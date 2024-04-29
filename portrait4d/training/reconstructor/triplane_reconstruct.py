# Reconstructor for Portrait4D, modified from EG3D: https://github.com/NVlabs/eg3d

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn.functional as F
import dnnlib
from torch_utils import persistence

from models.stylegan.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.generator.triplane import OSGDecoder
from training.reconstructor.networks_reconstructor import EncoderGlobal, EncoderDetail, EncoderCanonical, DecoderTriplane, EncoderBG
from training.deformer.deformation import DeformationModule, DeformationModuleOnlyHead
from training.deformer.deform_utils import cam_world_matrix_transform
from training.volumetric_rendering.renderer import DeformImportanceRendererNew
from training.volumetric_rendering.ray_sampler import RaySampler

# Animatable triplane reconstructor Psi in Portrait4D
@persistence.persistent_class
class TriPlaneReconstructorNeutralize(torch.nn.Module):
    def __init__(self,
        img_resolution = 512, 
        mot_dims = 512,
        w_dim = 512,
        sr_num_fp16_res     = 0,
        has_background = False,
        has_superresolution = True,
        flame_full = True,
        masked_sampling = False,
        num_blocks_neutral = 4,
        num_blocks_motion = 4,
        motion_map_layers = 2,
        neural_rendering_resolution = 64,
        deformation_kwargs  = {},
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        
        self.mot_dims = mot_dims
        self.motion_map_layers = motion_map_layers

        self.encoder_global = EncoderGlobal()
        self.encoder_detail = EncoderDetail()
        self.encoder_canonical = EncoderCanonical(num_blocks_neutral=num_blocks_neutral, num_blocks_motion=num_blocks_motion, mot_dims=mot_dims, mapping_layers=motion_map_layers)
        self.generator_triplane = DecoderTriplane()

        self.renderer = DeformImportanceRendererNew()
        self.masked_sampling = masked_sampling
        self.ray_sampler = RaySampler()

        self.has_background = has_background
        self.has_superresolution = has_superresolution
        decoder_output_dim = 32 if has_superresolution else 3
        self.img_resolution=img_resolution

        if self.has_background:
            self.background = EncoderBG()

        if self.has_superresolution:
            superres_module_name = rendering_kwargs['superresolution_module'].replace('training.superresolution','models.stylegan.superresolution')
            self.superresolution = dnnlib.util.construct_class_by_name(class_name=superres_module_name, channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        else:
            self.superresolution = None        
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': decoder_output_dim})
        self.neural_rendering_resolution = neural_rendering_resolution
        self.rendering_kwargs = rendering_kwargs

        self.deformer = DeformationModuleOnlyHead(flame_full=flame_full,**deformation_kwargs)
    
        self._last_planes = None

        self.register_buffer('w_avg', torch.zeros([w_dim]))
    
    def _warping(self,images,flows):
        # images [(B, M, C, H, W)]
        # flows (B, M, 2, H, W) # inverse warpping flow
        warp_images = []
        B, M, _, H_f, W_f = flows.shape
        flows = flows.view(B*M,2,H_f, W_f)

        for im in images:
            B, M, C, H, W = im.shape
            im = im.view(B*M, C, H, W)
            y, x = torch.meshgrid(torch.linspace(-1, 1, H, dtype=torch.float32, device=im.device), torch.linspace(-1, 1, W, dtype=torch.float32, device=im.device), indexing='ij')
            xy = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(B*M,1,1,1) #(B,H,W,2)
            if H_f != H:
                _flows = F.interpolate(flows,size=(H,W), mode='bilinear', align_corners=True)
            else:
                _flows = flows
            _flows = _flows.permute(0,2,3,1) #(B,H,W,2)
            uv = _flows + xy
            warp_image = F.grid_sample(im, uv, mode='bilinear', padding_mode='zeros', align_corners=True) #(B,C,H,W)
            warp_image = warp_image.view(B, M, C, H, W)
            warp_images.append(warp_image)

        return warp_images

    def _deformer(self,shape_params,exp_params,pose_params,eye_pose_params, use_rotation_limits=False, smooth_th=3e-3):
        return lambda coordinates: self.deformer(coordinates, shape_params,exp_params,pose_params,eye_pose_params, use_rotation_limits=use_rotation_limits, smooth_th=smooth_th)

    def synthesis(self, imgs_app, imgs_mot, motions_app, motions, c, _deformer, neural_rendering_resolution=None, neural_rendering_resolution_patch=None, cache_backbone=False, use_cached_backbone=False, use_rotation_limits=None, smpl_param=None, patch_scale=1.0, run_full=True, uv=None, chunk=None, flame_flow=None, motion_scale=1.0, box_warp_scale=1.0, **synthesis_kwargs):

        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        world2cam_matrix = cam_world_matrix_transform(cam2world_matrix)
        cam_z = world2cam_matrix[:,2,3]
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        N = cam2world_matrix.shape[0]

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        elif self.training:
            self.neural_rendering_resolution = neural_rendering_resolution

        if neural_rendering_resolution_patch is None:
            neural_rendering_resolution_patch = neural_rendering_resolution

        H = W = neural_rendering_resolution
        H_patch = W_patch = neural_rendering_resolution_patch
        
        # Create triplanes by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            features_global = self.encoder_global(imgs_app)
            features_detail = self.encoder_detail(imgs_app)
            
            features_canonical = self.encoder_canonical(features_global, motions, motions_app, scale=motion_scale)

            features_canonical_lr = features_canonical[0]
            features_canonical_sr = features_canonical[1]

            planes = self.generator_triplane(features_canonical_sr,features_detail)
            
            # Reshape output into three 32-channel planes
            if not isinstance(planes, list):
                planes = [planes]
            planes = [p.view(len(p), -1, 32, p.shape[-2], p.shape[-1]) for p in planes]

        warp_planes = None # deprecated

        if cache_backbone:
            self._last_planes = planes

        if self.has_background:
            background = self.background(imgs_app)
            background = torch.sigmoid(background)
            if background.shape[-1] != neural_rendering_resolution:
                background = F.interpolate(background,size=(neural_rendering_resolution,neural_rendering_resolution),mode='bilinear')
        else:
            background = 0

        # Create a batch of rays for volume rendering
        output = {}
        if run_full:
            ray_origins, ray_directions, _ = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
            # Perform volume rendering
            if chunk is None:
                feature_samples, depth_samples, all_depths, all_weights, T_bg = self.renderer((planes[0:1], planes[1:2])[neural_rendering_resolution>128], self.decoder, _deformer, ray_origins, ray_directions, self.rendering_kwargs, cam_z=cam_z, box_warp_scale=box_warp_scale) # channels last
            
            else:
                feature_samples, depth_samples, all_depths, all_weights, T_bg = list(), list(), list(), list(), list()
                for _ro, _rd in zip(torch.split(ray_origins, chunk, dim=1), torch.split(ray_directions, chunk, dim=1)):
                    _f, _d, _ad, _aw, _tbg = self.renderer((planes[0:1], planes[1:2])[neural_rendering_resolution>128], self.decoder, _deformer, _ro, _rd, self.rendering_kwargs, cam_z=cam_z, box_warp_scale=box_warp_scale) # channels last
                    feature_samples.append(_f)
                    depth_samples.append(_d)
                    all_depths.append(_ad)
                    all_weights.append(_aw)
                    T_bg.append(_tbg)

                feature_samples = torch.cat(feature_samples, 1)
                depth_samples = torch.cat(depth_samples, 1)
                all_depths = torch.cat(all_depths, 1)
                all_weights = torch.cat(all_weights, 1)
                T_bg = torch.cat(T_bg, 1)

            weights_samples = all_weights.sum(2)
            
            # Reshape into 'raw' neural-rendered image
            # H = W = self.neural_rendering_resolution
            feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
            depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

            # background
            if feature_image.shape[-1]<=128:
                T_bg = T_bg.permute(0, 2, 1).reshape(N, 1, H, W)
                feature_image_wobg = feature_image
                if self.has_background:
                    feature_image = feature_image + T_bg*background
                    rgb_bg = 2*background[:, :3] - 1
                    output.update({'rgb_bg': rgb_bg})
                else:
                    feature_image = feature_image + T_bg*0
                feature_image_wobg = 2*feature_image_wobg - 1
                feature_image = 2*feature_image - 1
                rgb_image_wobg = feature_image_wobg[:,:3]
                rgb_image = feature_image[:, :3]
                if self.superresolution is not None and rgb_image.shape[-1]<=128:
                    ws = self.w_avg.reshape(1,1,-1).repeat(N,1,1)
                    # ws = torch.ones([N,1,512]).to(rgb_image.device) following https://research.nvidia.com/labs/nxp/lp3d/
                    sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
                else:
                    sr_image = rgb_image
            else:
                rgb_image = feature_image[:, :3]
                if self.has_background:
                    ws = self.w_avg.reshape(1,1,-1).repeat(N,1,1)
                    background_sr = 2*background - 1
                    background_sr = self.superresolution(background_sr[:, :3], background_sr, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
                    background_sr = (background_sr + 1) * 0.5
                    T_bg = T_bg.permute(0, 2, 1).reshape(N, 1, H, W)
                    rgb_image = rgb_image + T_bg*background_sr
                else:
                    T_bg = T_bg.permute(0, 2, 1).reshape(N, 1, H, W)
                    rgb_image = rgb_image + T_bg*0
                rgb_image = 2*rgb_image - 1
                sr_image = rgb_image

            output.update({'image': rgb_image, 'image_wobg':rgb_image_wobg, 'image_feature': feature_image, 'image_feature_wobg': feature_image_wobg, 'background_feature': 2*background-1, 'image_sr':sr_image, 'image_depth': depth_image, 'interval':all_depths.squeeze(-1), 'all_weights':all_weights.squeeze(-1), \
                'T_bg': T_bg, 'seg': (1 - T_bg)*2 - 1, 'uv': uv, 'planes':planes, 'warp_planes':warp_planes})

        if patch_scale<1:
            with torch.no_grad():
                head_mask = self.deformer.renderer(smpl_param[0], smpl_param[1], smpl_param[2], smpl_param[3], c, half_size = int(self.img_resolution/2), only_face=False,cull_backfaces=False, use_rotation_limits=use_rotation_limits)[1]

            patch_ray_origins, patch_ray_directions, patch_info = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution_patch, patch_scale=patch_scale, mask=[head_mask, head_mask], masked_sampling=self.masked_sampling)
            if self.has_background:
                ws = self.w_avg.reshape(1,1,-1).repeat(N,1,1)
                # ws = torch.ones([N,1,512]).to(background_sr.device) # follow https://research.nvidia.com/labs/nxp/lp3d/
                background_sr = 2*background - 1
                background_sr = self.superresolution(background_sr[:, :3], background_sr, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
                background_sr = (background_sr+1)*0.5
                patch_background_sr = []

            if uv is not None:
                patch_uv = []
            if run_full:
                patch_sr_image = []
                patch_rgb_image = []
                sr_image_ = sr_image.detach()
                rgb_image_ = rgb_image.detach()
                rgb_image_ = torch.nn.functional.interpolate(rgb_image_, size=(sr_image_.shape[-1]),
                                mode='bilinear', align_corners=False, antialias=True)

            for i in range(len(patch_info)):
                top, left = patch_info[i]
                if uv is not None:
                    patch_uv.append(uv[i:i+1, :, top:top+neural_rendering_resolution_patch, left:left+neural_rendering_resolution_patch])
                if run_full:
                    patch_sr_image.append(sr_image_[i:i+1, :, top:top+neural_rendering_resolution_patch, left:left+neural_rendering_resolution_patch])
                    patch_rgb_image.append(rgb_image_[i:i+1, :, top:top+neural_rendering_resolution_patch, left:left+neural_rendering_resolution_patch])
                if self.has_background:
                    patch_background_sr.append(background_sr[i:i+1, :, top:top+neural_rendering_resolution_patch, left:left+neural_rendering_resolution_patch])

            if uv is not None:
                patch_uv = torch.cat(patch_uv, 0)
            else:
                patch_uv = None
            if run_full:
                patch_sr_image = torch.cat(patch_sr_image, 0)
                patch_rgb_image = torch.cat(patch_rgb_image, 0)
                output.update({'patch_image': patch_sr_image, 'patch_image_gr': patch_rgb_image})
            if self.has_background:
                patch_background_sr = torch.cat(patch_background_sr, 0)

            # Perform volume rendering
            patch_feature_samples, patch_depth_samples, patch_all_depths, patch_all_weights, patch_T_bg = self.renderer(planes[1:2], self.decoder, _deformer, patch_ray_origins, patch_ray_directions, self.rendering_kwargs, cam_z=cam_z, box_warp_scale=box_warp_scale) # channels last
            # Reshape into 'raw' neural-rendered image
            patch_feature_image = patch_feature_samples.permute(0, 2, 1).reshape(N, patch_feature_samples.shape[-1], H_patch, W_patch).contiguous()
            patch_depth_image = patch_depth_samples.permute(0, 2, 1).reshape(N, 1, H_patch, W_patch)
            patch_rgb_image = patch_feature_image[:, :3]

            if self.has_background:
                patch_T_bg = patch_T_bg.permute(0, 2, 1).reshape(N, 1, H_patch, W_patch)
                patch_rgb_image = patch_rgb_image + patch_T_bg * patch_background_sr
                patch_rgb_image = 2*patch_rgb_image-1

            output.update({'patch_image_raw': patch_rgb_image, 'patch_seg': (1 - patch_T_bg)*2 - 1, 'patch_T_bg': patch_T_bg, 'patch_uv': patch_uv, 'patch_all_depths': patch_all_depths.squeeze(-1), 'patch_all_weights': patch_all_weights.squeeze(-1)})

        return output


    def sample_mixed(self, imgs_app, imgs_mot, motions_app, motions, coordinates, directions, shape_params, exp_params, pose_params, eye_pose_params, use_deform=True, motion_scale=1.0,  **synthesis_kwargs):
        
        features_global = self.encoder_global(imgs_app)
        features_detail = self.encoder_detail(imgs_app)
        features_canonical = self.encoder_canonical(features_global, motions, motions_app, scale=motion_scale)
        features_canonical_lr = features_canonical[0]
        features_canonical_sr = features_canonical[1]
        planes = self.generator_triplane(features_canonical_sr,features_detail)

        if not isinstance(planes, list):
            planes = [planes]
        planes = [p.view(len(p), -1, 32, p.shape[-2], p.shape[-1]) for p in planes]  

        # target space to canonical space deformation
        if use_deform:
            _deformer = self._deformer(shape_params,exp_params,pose_params,eye_pose_params)
            out_deform = _deformer(coordinates)
            coordinates = out_deform['canonical']
        else:
            coordinates = coordinates    

        out = self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)
        out['canonical'] = coordinates

        return out

    def get_planes(self, imgs_app, imgs_mot, motions_app, motions, coordinates, directions, shape_params, exp_params, pose_params, eye_pose_params, use_deform=True, motion_scale=1.0, **synthesis_kwargs):
        
        features_global = self.encoder_global(imgs_app)
        features_detail = self.encoder_detail(imgs_app)
        features_canonical = self.encoder_canonical(features_global, motions, motions_app, scale=motion_scale)
        features_canonical_lr = features_canonical[0]
        features_canonical_sr = features_canonical[1]
        planes = self.generator_triplane(features_canonical_sr,features_detail)

        out = {}
        out['planes'] = planes

        return out

    def forward(self, imgs_app, imgs_mot, motions_app, motions, shape_params, exp_params, pose_params, eye_pose_params, c, neural_rendering_resolution=None, neural_rendering_resolution_patch=None, cache_backbone=False, use_cached_backbone=False, patch_scale=1.0, motion_scale=1.0, **synthesis_kwargs):
        # Render a batch of generated images.
        _deformer = self._deformer(shape_params,exp_params,pose_params,eye_pose_params)

        uv = self.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, c, half_size=int(self.img_resolution/2))[0]

        img = self.synthesis(imgs_app, imgs_mot, motions_app, motions, c, _deformer=_deformer, neural_rendering_resolution=neural_rendering_resolution, neural_rendering_resolution_patch=neural_rendering_resolution_patch, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, \
         motion_scale=motion_scale, smpl_param=(shape_params, exp_params, pose_params, eye_pose_params), patch_scale=patch_scale, **synthesis_kwargs)
        img['uv'] = uv

        return img