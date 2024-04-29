# Generator for GenHead, modified from EG3D: https://github.com/NVlabs/eg3d

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
from torch import nn
from torch_utils import persistence
from models.stylegan.networks_stylegan2 import Generator as StyleGAN2Backbone
from models.stylegan.networks_stylegan2 import ToRGBLayer, FullyConnectedLayer, SynthesisNetwork
from models.stylegan.superresolution import SuperresolutionPatchMLP
from training.deformer.deformation import DeformationModule, DeformationModuleOnlyHead
from training.deformer.deform_utils import cam_world_matrix_transform
from training.volumetric_rendering.renderer import ImportanceRenderer, DeformImportanceRenderer, PartDeformImportanceRenderer, DeformImportanceRendererNew
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
import torch.nn.functional as F
from torch_utils.ops import upfirdn2d
import copy

# Baseline generator without separate deformation for face, eyes, and mouth
@persistence.persistent_class
class TriPlaneGeneratorDeform(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        deformation_kwargs  = {},
        sr_kwargs = {},
        has_background = True,
        has_superresolution = False,
        flame_condition = False,
        flame_full = False,
        dynamic_texture = False, # Deprecated
        random_combine = True,
        triplane_resolution = 256,
        triplane_channels = 96,
        masked_sampling = None,
        has_patch_sr = False,
        add_block = False,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.flame_condition = flame_condition
        self.has_background = has_background
        self.has_superresolution = has_superresolution
        self.dynamic_texture = dynamic_texture
        decoder_output_dim = 32 if has_superresolution else 3
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = DeformImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)

        self.to_dynamic = None

        if self.has_background:
            self.background = StyleGAN2Backbone(z_dim, 0, w_dim, img_resolution=64, mapping_kwargs={'num_layers':8}, channel_base=16384, img_channels=decoder_output_dim)

        if self.has_superresolution:
            self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        else:
            self.superresolution = None     
               
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': decoder_output_dim})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs

        self.deformer = DeformationModule(flame_full=flame_full,dynamic_texture=dynamic_texture, **deformation_kwargs)
    
        self._last_planes = None

    def _deformer(self,shape_params,exp_params,pose_params,eye_pose_params,ws,c_deform,cache_backbone=False, use_cached_backbone=False, use_rotation_limits=False, eye_blink_params=None):
        return lambda coordinates: self.deformer(coordinates, shape_params,exp_params,pose_params,eye_pose_params,ws,c_deform,cache_backbone=cache_backbone,use_cached_backbone=use_cached_backbone, use_rotation_limits=use_rotation_limits)
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, z_bg, c, _deformer, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, use_dynamic=False,use_rotation_limits=None, smpl_param=None, patch_scale=None, chunk=None, run_full=None, uv=None, diff_dynamic=False, forward_mode='train', eye_blink_params=None, ws_super=None, **synthesis_kwargs):
        
        if forward_mode == 'train':
            face_ws = ws
            dynamic_ws = ws
        # for inversion only
        elif ws.shape[1] >= self.backbone.num_ws + self.background.num_ws:
            face_ws, bg_ws, dynamic_ws = ws[:, :self.backbone.num_ws, :], ws[:, self.backbone.num_ws:self.backbone.num_ws+self.background.num_ws, :], ws[:, self.backbone.num_ws+self.background.num_ws:, :]
        else:
            face_ws, bg_ws, dynamic_ws = ws[:, :self.backbone.num_ws, :], ws[:, self.backbone.num_ws:-1, :], ws[:, self.backbone.num_ws-1:self.backbone.num_ws, :]

        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        world2cam_matrix = cam_world_matrix_transform(cam2world_matrix)
        cam_z = world2cam_matrix[:,2,3]
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions, _ = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)


        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes, last_featuremap = self.backbone.synthesis(face_ws, update_emas=update_emas, **synthesis_kwargs)

        if cache_backbone:
            self._last_planes = planes
        
        planes = planes.view(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])

        if self.dynamic_texture:
            pass # deprecated
        else:
            dynamic_planes = None


        # Perform volume rendering
        feature_samples, depth_samples, all_depths, all_weights, T_bg, offset, dist_to_surface, vts_mask, vts_mask_region, coarse_sample_points, coarse_triplane_features = self.renderer(planes, self.decoder, _deformer, ray_origins, ray_directions, self.rendering_kwargs, dynamic=self.dynamic_texture, cam_z=cam_z) # channels last

        weights_samples = all_weights.sum(2)
        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # background
        if self.has_background:
            if forward_mode == 'train':
                background = self.background(z_bg, c, **synthesis_kwargs)
            else:
                background, _ = self.background.synthesis(bg_ws, update_emas=update_emas, **synthesis_kwargs)
            background = torch.sigmoid(background) # (-1,1) (N,3,H,W)

            if background.shape[-1] != neural_rendering_resolution:
                background = F.interpolate(background,size=(neural_rendering_resolution,neural_rendering_resolution),mode='bilinear')
                
            T_bg = T_bg.permute(0, 2, 1).reshape(N, 1, H, W)
            feature_image = feature_image + T_bg*background

        else:
            T_bg = T_bg.permute(0, 2, 1).reshape(N, 1, H, W)
            background = 0.
        
        feature_image = 2*feature_image - 1
        rgb_image = feature_image[:, :3]
        
        if self.superresolution is not None:
            sr_image = self.superresolution(rgb_image, feature_image, face_ws, ws_super=ws_super, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_image = None

        return {'image': rgb_image, 'image_feature':feature_image, 'image_sr':sr_image, 'image_depth': depth_image, 'background':2*background-1, 'interval':all_depths.squeeze(-1), 'all_weights':all_weights.squeeze(-1), 'T_bg': T_bg, \
            'seg': (1 - T_bg)*2 - 1, 'offset':offset, 'dist_to_surface':dist_to_surface, 'vts_mask':vts_mask, 'vts_mask_region':vts_mask_region, 'dynamic_planes':dynamic_planes, 'coarse_sample_points':coarse_sample_points, 'coarse_triplane_features':coarse_triplane_features}
    
    
    def sample(self, coordinates, directions, shape_params,exp_params,pose_params,eye_pose_params, z, c, use_deform=True, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes, last_featuremap = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])

        # target space to canonical space deformation
        if use_deform:
            _deformer = self._deformer(shape_params,exp_params,pose_params,eye_pose_params)
            out_deform = _deformer(coordinates)
            coordinates = out_deform['canonical']
            offset = out_deform['offset']
            dynamic_mask = out_deform['dynamic_mask']
        else:
            coordinates = coordinates
            offset = torch.zeros_like(coordinates)
            dynamic_mask = None

        out = self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, dynamic_mask=dynamic_mask)
        out['canonical'] = coordinates
        out['offset'] = offset

        return out

    def sample_mixed(self, coordinates, directions, shape_params,exp_params,pose_params,eye_pose_params, ws, use_deform=True, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes, last_featuremap = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        # planes = torch.tanh(planes)
        planes = planes.view(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])

        # target space to canonical space deformation
        if use_deform:
            _deformer = self._deformer(shape_params,exp_params,pose_params,eye_pose_params)
            out_deform = _deformer(coordinates)
            coordinates = out_deform['canonical']
            offset = out_deform['offset']
            dynamic_mask = out_deform['dynamic_mask']
        else:
            coordinates = coordinates
            offset = torch.zeros_like(coordinates)  
            dynamic_mask = None          

        out = self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, dynamic_mask=dynamic_mask)
        out['canonical'] = coordinates
        out['offset'] = offset

        return out

    def forward(self, shape_params,exp_params,pose_params,eye_pose_params, z, z_bg, c, c_compose, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, patch_scale=None, **synthesis_kwargs):    
        # Render a batch of generated images.
        _deformer = self._deformer(shape_params,exp_params,pose_params,eye_pose_params)
        
        c_compose_condition = c_compose.clone()
        if self.flame_condition:
            c_compose_condition = torch.cat([c_compose_condition,shape_params,exp_params],dim=-1)
        
        ws = self.mapping(z, c_compose_condition, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        
        # Render correspondence map as condition to the discriminator
        uv = self.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, c, half_size=int(self.img_resolution/2))[0]

        img = self.synthesis(ws, z_bg, c, _deformer=_deformer, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        img['uv'] = uv

        return img

# Generator used in GenHead, with part-wise deformation
@persistence.persistent_class
class PartTriPlaneGeneratorDeform(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        triplane_channels,
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        deformation_kwargs  = {},
        sr_kwargs = {},
        has_background = True,
        has_superresolution = False,
        has_patch_sr = False,
        flame_condition = True,
        flame_full = False,
        dynamic_texture = False, # Deprecated
        random_combine = True,
        add_block = False,
        triplane_resolution = 256,
        masked_sampling = False,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.flame_condition = flame_condition
        self.dynamic_texture = dynamic_texture
        self.has_background = has_background
        self.has_superresolution = has_superresolution
        self.has_patch_sr = has_patch_sr
        decoder_output_dim = 32 if has_superresolution else 3
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = PartDeformImportanceRenderer() if triplane_channels>96 else DeformImportanceRenderer()
        self.mouth_part = True #if triplane_channels>192 else False
        self.mouth_dynamic = False
        self.masked_sampling = masked_sampling
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=triplane_resolution, img_channels=triplane_channels, mapping_kwargs=mapping_kwargs, add_block=add_block, **synthesis_kwargs)
        if self.has_background:
            self.background = StyleGAN2Backbone(z_dim, 0, w_dim, img_resolution=64, mapping_kwargs={'num_layers':8}, channel_base=16384, img_channels=decoder_output_dim)
        if self.has_superresolution:
            self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        else:
            self.superresolution = None        
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': decoder_output_dim})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs

        if self.has_patch_sr:
            self.patch_sr = SuperresolutionPatchMLP(channels=32, img_resolution=None, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=True)

        self.to_dynamic = None
        self.to_dynamic_sr = None

        self.deformer = DeformationModule(flame_full=flame_full,dynamic_texture=dynamic_texture,part=True,**deformation_kwargs)
    
        self._last_planes = None
        self._last_dynamic_planes = None
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

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
            xy = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(B,1,1,1) #(B,H,W,2)
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

    def _deformer(self,shape_params,exp_params,pose_params,eye_pose_params,eye_blink_params=None,exp_params_dynamics=None,cache_backbone=False, use_cached_backbone=False,use_rotation_limits=False):
        return lambda coordinates, mouth: self.deformer(coordinates, shape_params,exp_params,pose_params,eye_pose_params,eye_blink_params=eye_blink_params,exp_params_dynamics=exp_params_dynamics,cache_backbone=cache_backbone,use_cached_backbone=use_cached_backbone,use_rotation_limits=use_rotation_limits,mouth=mouth)
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, z_bg, c, _deformer, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, use_dynamic=False, use_rotation_limits=None, smpl_param=None, eye_blink_params=None, patch_scale=1.0, run_full=True, uv=None, chunk=None, diff_dynamic=False, dense_eye=False, forward_mode='train', ws_super=None, **synthesis_kwargs):
        
        if forward_mode == 'train':
            face_ws = ws
            dynamic_ws = ws
        # for inversion only
        elif ws.shape[1] >= self.backbone.num_ws + self.background.num_ws:
            face_ws, bg_ws, dynamic_ws = ws[:, :self.backbone.num_ws, :], ws[:, self.backbone.num_ws:self.backbone.num_ws+self.background.num_ws, :], ws[:, self.backbone.num_ws+self.background.num_ws:, :]
        else:
            face_ws, bg_ws, dynamic_ws = ws[:, :self.backbone.num_ws, :], ws[:, self.backbone.num_ws:-1, :], ws[:, self.backbone.num_ws-1:self.backbone.num_ws, :]
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        world2cam_matrix = cam_world_matrix_transform(cam2world_matrix)
        cam_z = world2cam_matrix[:,2,3]
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        N = cam2world_matrix.shape[0]

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        elif self.training:
            self.neural_rendering_resolution = neural_rendering_resolution
        H = W = neural_rendering_resolution
        
        with torch.no_grad():
            eye_mask = self.deformer.renderer(smpl_param[0], smpl_param[1], smpl_param[2], smpl_param[3], c, half_size = int(self.img_resolution/2), eye_blink_params=eye_blink_params, eye_mask=True, use_rotation_limits=use_rotation_limits)[1]
            face_wo_eye_mask = self.deformer.renderer(smpl_param[0], smpl_param[1], smpl_param[2], smpl_param[3], c, half_size = int(self.img_resolution/2), eye_blink_params=eye_blink_params, face_woeye=True, use_rotation_limits=use_rotation_limits)[1]
            eye_mask = eye_mask * (1-face_wo_eye_mask)
            blur_sigma = 1
            blur_size = blur_sigma * 3
            f = torch.arange(-blur_size, blur_size + 1, device=eye_mask.device).div(blur_sigma).square().neg().exp2()
            eye_mask_sr = upfirdn2d.filter2d(eye_mask, f / f.sum())
            eye_mask = torch.nn.functional.interpolate(eye_mask, size=(self.neural_rendering_resolution), mode='bilinear', align_corners=False, antialias=True)
            head_mask = self.deformer.renderer(smpl_param[0], smpl_param[1], smpl_param[2], smpl_param[3], c, half_size = int(self.img_resolution/2), eye_blink_params=eye_blink_params, only_face=False,cull_backfaces=False, use_rotation_limits=use_rotation_limits)[1]
            if self.mouth_part:
                head_wo_mouth_mask = self.deformer.renderer(smpl_param[0], smpl_param[1], smpl_param[2], smpl_param[3], c, half_size = int(self.img_resolution/2), eye_blink_params=eye_blink_params, only_face=False,cull_backfaces=True, noinmouth=True, use_rotation_limits=use_rotation_limits)[1]
                mouth_mask = head_mask * (1-head_wo_mouth_mask)
                mouth_mask_sr = -self.max_pool(-mouth_mask)
                mouth_mask_sr = self.max_pool(mouth_mask_sr)
                blur_sigma = 2
                blur_size = blur_sigma * 3
                f = torch.arange(-blur_size, blur_size + 1, device=mouth_mask_sr.device).div(blur_sigma).square().neg().exp2()
                mouth_mask_sr = upfirdn2d.filter2d(mouth_mask_sr, f / f.sum())
                mouth_mask = torch.nn.functional.interpolate(mouth_mask, size=(self.neural_rendering_resolution), mode='bilinear', align_corners=False, antialias=True)
                mouth_mask_sr = (mouth_mask_sr + torch.nn.functional.interpolate(mouth_mask, size=(self.img_resolution), mode='bilinear', align_corners=False, antialias=True)).clamp(max=1)
                # mouth_mask_sr[:,:,:128,:] *= 0  # for visualization only (deprecated)
            else:
                mouth_mask = None
                mouth_mask_sr = None
            head_mask = torch.nn.functional.interpolate(head_mask, size=(neural_rendering_resolution), mode='bilinear', align_corners=False, antialias=True)
            head_mask_sr = torch.nn.functional.interpolate(head_mask, size=(self.img_resolution), mode='bilinear', align_corners=False, antialias=True)
        
        # Create triplanes by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
            dynamic_planes = self._last_dynamic_planes
        else:
            planes, last_featuremap = self.backbone.synthesis(face_ws, update_emas=update_emas, **synthesis_kwargs)

            # Reshape output into three 32-channel planes
            if not isinstance(planes, list):
                planes = [planes]
                last_featuremap = [last_featuremap]
            planes = [p.view(len(p), -1, 32, p.shape[-2], p.shape[-1]) for p in planes]

            if self.dynamic_texture:
                pass # deprecated
            else:
                dynamic_planes = None

        if cache_backbone:
            self._last_planes = planes
            self._last_dynamic_planes = dynamic_planes

        if self.has_background:
            if forward_mode == 'train':
                background = self.background(z_bg, c, **synthesis_kwargs)
            else:
                background, _ = self.background.synthesis(bg_ws, update_emas=update_emas, **synthesis_kwargs)
            background = torch.sigmoid(background) # (-1,1) (N,3,H,W)
            background_feature = F.interpolate(background,size=(self.img_resolution,self.img_resolution),mode='bilinear')
            if background.shape[-1] != neural_rendering_resolution:
                background = F.interpolate(background,size=(neural_rendering_resolution,neural_rendering_resolution),mode='bilinear')
                
        # Create a batch of rays for volume rendering
        output = {}
        coarse_sample_points = coarse_triplane_features = None
        if run_full:
            ray_origins, ray_directions, _ = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
            mouth_mask_flat = mouth_mask[:, 0].view(mouth_mask.shape[0], -1, 1) if neural_rendering_resolution==self.neural_rendering_resolution else mouth_mask_sr[:, 0].view(mouth_mask_sr.shape[0], -1, 1)
            eye_mask_flat = eye_mask[:, 0].view(eye_mask.shape[0], -1, 1) if neural_rendering_resolution==self.neural_rendering_resolution else eye_mask_sr[:, 0].view(eye_mask_sr.shape[0], -1, 1)
            # Perform volume rendering
            if chunk is None:
                feature_samples, depth_samples, all_depths, all_weights, T_bg, offset, dist_to_surface, densities_face_ineye, densities_face_inmouth,vts_mask, vts_mask_region, coarse_sample_points, coarse_triplane_features, eye_mask_sel, mouth_mask_sel \
                    = self.renderer((planes[0:1], planes)[neural_rendering_resolution>64], self.decoder, _deformer, ray_origins, ray_directions, self.rendering_kwargs, eye_mask=eye_mask_flat, mouth_mask=mouth_mask_flat, mouth_dynamic=self.mouth_dynamic, auto_fuse=(False, True)[neural_rendering_resolution>64],cam_z=cam_z) # channels last
                if dense_eye: # only for batchsize=1
                    is_eye_region = (eye_mask_flat!=0).squeeze(0).squeeze(-1)
                    if torch.sum(is_eye_region.to(torch.float32)) == 0:
                        feature_samples_eye = 0
                        T_bg_eye = 0
                    else:
                        ray_origins_eye = ray_origins[:,is_eye_region]
                        ray_directions_eye = ray_directions[:,is_eye_region]
                        eye_mask_eye = eye_mask_flat[:,is_eye_region]

                        rendering_kwargs_eye = copy.deepcopy(self.rendering_kwargs)
                        rendering_kwargs_eye['depth_resolution'] = 128
                        rendering_kwargs_eye['depth_resolution_importance'] = 128

                        feature_samples_eye, depth_samples_eye, all_depths_eye, all_weights_eye, T_bg_eye, offset_eye, dist_to_surface_eye, densities_face_ineye_eye, densities_face_inmouth_eye,vts_mask_eye, vts_mask_region_eye, _, _, _, _ = self.renderer((planes[0:1], planes)[neural_rendering_resolution>64], self.decoder, _deformer, ray_origins_eye, ray_directions_eye, rendering_kwargs_eye, eye_mask=eye_mask_eye, mouth_mask=None, mouth_dynamic=self.mouth_dynamic, auto_fuse=(False, True)[neural_rendering_resolution>64]) # channels last
            
            else:
                feature_samples, depth_samples, all_depths, all_weights, T_bg, offset, dist_to_surface, densities_face_ineye, densities_face_inmouth, vts_mask, vts_mask_region = list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
                for _ro, _rd, _em, _mm in zip(torch.split(ray_origins, chunk, dim=1), torch.split(ray_directions, chunk, dim=1), torch.split(eye_mask_flat, chunk, dim=1), torch.split(mouth_mask_flat, chunk, dim=1)):
                    _f, _d, _ad, _aw, _tbg, _off, _ds, _dfe, _dfm, _vm, _vmr = self.renderer((planes[0:1], planes)[neural_rendering_resolution>64], self.decoder, _deformer, _ro, _rd, self.rendering_kwargs, eye_mask=_em, mouth_mask=_mm, mouth_dynamic=self.mouth_dynamic, auto_fuse=(False, True)[neural_rendering_resolution>64],cam_z=cam_z) # channels last
                    feature_samples.append(_f)
                    depth_samples.append(_d)
                    all_depths.append(_ad)
                    all_weights.append(_aw)
                    T_bg.append(_tbg)
                    offset.append(_off)
                    dist_to_surface.append(_ds)
                    densities_face_ineye.append(_dfe)
                    densities_face_inmouth.append(_dfm)
                    vts_mask.append(_vm)
                    vts_mask_region.append(_vmr)

                feature_samples = torch.cat(feature_samples, 1)
                depth_samples = torch.cat(depth_samples, 1)
                all_depths = torch.cat(all_depths, 1)
                all_weights = torch.cat(all_weights, 1)
                T_bg = torch.cat(T_bg, 1)
                offset = torch.cat(offset, 1)
                dist_to_surface = torch.cat(dist_to_surface, 1)
                densities_face_ineye = torch.cat(densities_face_ineye, 1)
                densities_face_inmouth = torch.cat(densities_face_inmouth, 1)
                vts_mask = torch.cat(vts_mask, 1)
                vts_mask_region = torch.cat(vts_mask_region, 1)

            weights_samples = all_weights.sum(2)

            if dense_eye:
                feature_samples[:,is_eye_region] = feature_samples_eye
                T_bg[:,is_eye_region] = T_bg_eye

            # Reshape into 'raw' neural-rendered image
            # H = W = self.neural_rendering_resolution
            feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
            depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

            # background
            if feature_image.shape[-1]<=128:
                T_bg = T_bg.permute(0, 2, 1).reshape(N, 1, H, W)
                if self.has_background:
                    feature_image = feature_image + T_bg*background
                else:
                    feature_image = feature_image + T_bg
                feature_image = 2*feature_image - 1
                rgb_image = feature_image[:, :3]
                if self.superresolution is not None and rgb_image.shape[-1]<=128:
                    sr_image = self.superresolution(rgb_image, feature_image, ws, ws_super=ws_super, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
                else:
                    sr_image = rgb_image
            else:
                rgb_image = feature_image[:, :3]
                if self.has_background:
                    background_sr = 2*background - 1
                    background_sr = self.superresolution(background_sr[:, :3], background_sr, ws, ws_super=ws_super, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
                    background_sr = (background_sr + 1) * 0.5
                    T_bg = T_bg.permute(0, 2, 1).reshape(N, 1, H, W)
                    rgb_image = rgb_image + T_bg*background_sr
                else:
                    T_bg = T_bg.permute(0, 2, 1).reshape(N, 1, H, W)
                    rgb_image = rgb_image + T_bg
                rgb_image = 2*rgb_image - 1
                sr_image = rgb_image
                
                if self.has_patch_sr:
                    rgb_image_ = rgb_image
                    background_feature = torch.cat([background_sr,background_feature[:,3:]],dim=1)
                    feature_image_ = feature_image + T_bg * background_feature
                    feature_image_ = 2*feature_image_ - 1
                    sr_rgb_image = self.patch_sr(rgb_image_[:, :3], feature_image_, torch.ones_like(ws))
                    output.update({'image_raw_sr': sr_rgb_image})

            output.update({'image': rgb_image, 'image_feature':feature_image,'image_sr':sr_image, 'background':2*background-1, 'image_depth': depth_image, 'interval':all_depths.squeeze(-1), 'all_weights':all_weights.squeeze(-1), \
                'T_bg': T_bg, 'offset':offset, 'dist_to_surface':dist_to_surface, 'eye_mask': eye_mask, 'head_mask': head_mask, 'mouth_mask': mouth_mask, 'mouth_mask_sr': mouth_mask_sr+eye_mask_sr, \
                    'densities_face_ineye': densities_face_ineye, 'densities_face_inmouth': densities_face_inmouth, 'seg': (1 - T_bg)*2 - 1, 'vts_mask':vts_mask, 'vts_mask_region':vts_mask_region, 'dynamic_planes':dynamic_planes, 'uv': uv,\
                    'coarse_sample_points':coarse_sample_points, 'coarse_triplane_features':coarse_triplane_features, 'eye_mask_sel':eye_mask_sel, 'mouth_mask_sel':mouth_mask_sel})
        if patch_scale<1:
            patch_ray_origins, patch_ray_directions, patch_info = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution, patch_scale=patch_scale, mask=[eye_mask_sr+mouth_mask_sr, head_mask_sr], masked_sampling=self.masked_sampling)
            if self.has_background:
                background_sr = 2*background - 1
                background_sr = self.superresolution(background_sr[:, :3], background_sr, ws, ws_super=ws_super, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
                background_sr = (background_sr+1)*0.5
                patch_background_sr = []
                patch_background_feature = []
            patch_eye_mask = []
            if uv is not None:
                patch_uv = []
            if run_full:
                patch_sr_image = []
                patch_rgb_image = []
                sr_image_ = sr_image.detach()
                rgb_image_ = rgb_image.detach()
                rgb_image_ = torch.nn.functional.interpolate(rgb_image_, size=(sr_image_.shape[-1]),
                                mode='bilinear', align_corners=False, antialias=True)
            if self.mouth_part:
                patch_mouth_mask = []

            for i in range(len(patch_info)):
                top, left = patch_info[i]
                patch_eye_mask.append(eye_mask_sr[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                if uv is not None:
                    patch_uv.append(uv[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                if run_full:
                    patch_sr_image.append(sr_image_[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                    patch_rgb_image.append(rgb_image_[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                if self.mouth_part:
                    patch_mouth_mask.append(mouth_mask_sr[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                if self.has_background:
                    patch_background_sr.append(background_sr[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                    patch_background_feature.append(background_feature[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])

            patch_eye_mask = torch.cat(patch_eye_mask, 0)
            if uv is not None:
                patch_uv = torch.cat(patch_uv, 0)
            else:
                patch_uv = None
            if run_full:
                patch_sr_image = torch.cat(patch_sr_image, 0)
                patch_rgb_image = torch.cat(patch_rgb_image, 0)
                output.update({'patch_image': patch_sr_image, 'patch_image_gr': patch_rgb_image})
            if self.mouth_part:
                patch_mouth_mask = torch.cat(patch_mouth_mask, 0)
            if self.has_background:
                patch_background_sr = torch.cat(patch_background_sr, 0)
                patch_background_feature = torch.cat(patch_background_feature, 0)

            # Perform volume rendering
            patch_mouth_mask_flat = patch_mouth_mask[:, 0].view(mouth_mask.shape[0], -1, 1)
            patch_eye_mask_flat = patch_eye_mask[:, 0].view(eye_mask.shape[0], -1, 1)
            patch_feature_samples, patch_depth_samples, patch_all_depths, patch_all_weights, patch_T_bg, patch_offset, patch_dist_to_surface, patch_densities_face_ineye, patch_densities_face_inmouth, patch_vts_mask, patch_vts_mask_region, _, _, _, _ = self.renderer(planes, self.decoder, _deformer, patch_ray_origins, patch_ray_directions, self.rendering_kwargs, eye_mask=patch_eye_mask_flat, mouth_mask=patch_mouth_mask_flat, mouth_dynamic=self.mouth_dynamic, auto_fuse=True,cam_z=cam_z) # channels last
            # Reshape into 'raw' neural-rendered image
            patch_feature_image = patch_feature_samples.permute(0, 2, 1).reshape(N, patch_feature_samples.shape[-1], H, W).contiguous()
            patch_depth_image = patch_depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
            patch_rgb_image = patch_feature_image[:, :3]

            if self.has_background:
                patch_T_bg = patch_T_bg.permute(0, 2, 1).reshape(N, 1, H, W)
                patch_rgb_image = patch_rgb_image + patch_T_bg * patch_background_sr
                patch_rgb_image = 2*patch_rgb_image-1
            
            if self.has_patch_sr:
                patch_rgb_image_ = patch_rgb_image.clone().detach()
                patch_background_feature = torch.cat([patch_background_sr,patch_background_feature[:,3:]],dim=1)
                patch_feature_image_ = patch_feature_image.clone().detach() + patch_T_bg.clone().detach() * patch_background_feature.clone().detach()
                patch_feature_image_ = 2*patch_feature_image_ - 1
                sr_patch_rgb_image = self.patch_sr(patch_rgb_image_[:, :3], patch_feature_image_, torch.ones_like(ws))
                output.update({'patch_image_raw_sr': sr_patch_rgb_image})

            output.update({'patch_image_raw': patch_rgb_image, 'patch_seg': (1 - patch_T_bg)*2 - 1, 'patch_T_bg': patch_T_bg, 'patch_uv': patch_uv, 'patch_mouth_mask': patch_mouth_mask, 'patch_all_depths': patch_all_depths.squeeze(-1), 'patch_all_weights': patch_all_weights.squeeze(-1)})

        return output

    def sample(self, coordinates, directions, shape_params,exp_params,pose_params,eye_pose_params, z, c, use_deform=True, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        # planes = torch.tanh(planes)
        planes = planes.view(len(planes), -1, 32, planes.shape[-2], planes.shape[-1])

        # target space to canonical space deformation
        if use_deform:
            _deformer = self._deformer(shape_params,exp_params,pose_params,eye_pose_params)
            out_deform = _deformer(coordinates)
            coordinates = out_deform['canonical']
            offset = out_deform['offset']
        else:
            coordinates = coordinates
            offset = torch.zeros_like(coordinates)

        out = self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)
        out['canonical'] = coordinates
        out['offset'] = offset
        # out['offset'] = torch.zeros_like(coordinates)
        

        return out

    def sample_mixed(self, coordinates, directions, shape_params,exp_params,pose_params,eye_pose_params, ws, use_deform=True, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes, last_featuremap = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        # planes = planes.view(len(planes), -1, 32, planes.shape[-2], planes.shape[-1])
        if not isinstance(planes, list):
            planes = [planes]
            last_featuremap = [last_featuremap]
        planes = [p.view(len(p), -1, 32, p.shape[-2], p.shape[-1]) for p in planes]
        
        dynamic_planes = None

        # target space to canonical space deformation
        if use_deform:
            _deformer = self._deformer(shape_params,exp_params,pose_params,eye_pose_params)
            out_deform = _deformer(coordinates, mouth=self.mouth_part)
            coordinates_eye = out_deform['canonical_eye']
            coordinates_face = out_deform['canonical_face']
            coordinates_mouth = out_deform['canonical_mouth']
            if out_deform['dynamic_mask'] is not None:
                dynamic_mask = out_deform['dynamic_mask'] * (1-out_deform['inside_bbox_eye'][..., None])
            else:
                dynamic_mask = out_deform['dynamic_mask']
            offset = torch.zeros_like(coordinates_eye)
        else:
            coordinates = coordinates
            offset = torch.zeros_like(coordinates)
            dynamic_mask = None        

        plane_eye = [p[:, :3] for p in planes]
        if dynamic_mask is not None:
            if self.mouth_part and self.mouth_dynamic:
                plane_mouth = [torch.cat([p[:, :3],p[:, -3:]],dim=2) for p in planes]
            else:
                plane_mouth = [p[:, :3] for p in planes]
            plane_face = [torch.cat([p[:, 3:6],p[:, -3:]],dim=2) for p in planes]
        else:
            if self.mouth_part:
                plane_mouth = [p[:, :3] for p in planes]
            plane_face = [p[:, 3:6] for p in planes]

        out_eye = self.renderer.run_model(plane_eye, self.decoder, coordinates_eye, directions, self.rendering_kwargs,dynamic_mask=None)
        out_face = self.renderer.run_model(plane_face, self.decoder, coordinates_face, directions, self.rendering_kwargs,dynamic_mask=dynamic_mask)
        out_eye['canonical'] = coordinates_eye
        out_face['canonical'] = coordinates_face
        out_eye['offset'] = offset
        out_face['offset'] = offset
        if self.mouth_part:       
            out_mouth = self.renderer.run_model(plane_mouth, self.decoder, coordinates_mouth, directions, self.rendering_kwargs,dynamic_mask=(None, dynamic_mask)[self.mouth_dynamic])
            out_mouth['canonical'] = coordinates_mouth
            out_mouth['offset'] = offset
            return out_eye, out_face, out_mouth
        else:
            return out_eye, out_face

    def forward(self, shape_params,exp_params,pose_params,eye_pose_params, z, z_bg, c, c_compose, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, patch_scale=1.0, **synthesis_kwargs):
        # Render a batch of generated images.
        _deformer = self._deformer(shape_params,exp_params,pose_params,eye_pose_params)
        
        c_compose_condition = c_compose.clone()
        if self.flame_condition:
            c_compose_condition = torch.cat([c_compose_condition,shape_params],dim=-1)

        ws = self.mapping(z, c_compose_condition, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        
        # Render correspondence map as condition to the discriminator
        render_out = self.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, c, half_size=int(self.img_resolution/2))
        uv = render_out[0]
        landmarks = render_out[-1]        

        img = self.synthesis(ws, z_bg, c, _deformer=_deformer, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, smpl_param=(shape_params, exp_params, pose_params, eye_pose_params), patch_scale=patch_scale, **synthesis_kwargs)
        img['uv'] = uv
        img['landmarks'] = landmarks

        return img


def zero_init(m):
    with torch.no_grad():
        nn.init.constant_(m.weight,0)
        nn.init.constant_(m.bias,0)

@persistence.persistent_class
class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus()
        )
        self.out_sigma = FullyConnectedLayer(self.hidden_dim, 1, lr_multiplier=options['decoder_lr_mul'])
        self.out_rgb = FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])

        self.out_sigma.apply(zero_init)
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        rgb = self.out_rgb(x)
        sigma = self.out_sigma(x)

        rgb = rgb.view(N, M, -1)
        sigma = sigma.view(N, M, -1)

        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        return {'rgb': rgb, 'sigma': sigma}

