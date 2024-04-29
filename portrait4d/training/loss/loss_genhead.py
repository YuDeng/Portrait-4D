# Loss for GenHead, modified from EG3D: https://github.com/NVlabs/eg3d

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.discriminator.dual_discriminator import filtered_resizing
from training.loss.loss_utils import lossfun_distortion

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class PortraitSynthesisLoss(Loss):
    def __init__(self, device, G, D, D_patch=None, augment_pipe=None, lpips=None, r1_gamma=10, r1_gamma_patch=10, r1_gamma_uv=30, r1_gamma_seg=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_init_sigma_patch=0, blur_fade_kimg=0, blur_patch_seg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased', patch_scale=1.0, patch_gan=0.2, masked_sampling=None):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.D_patch            = D_patch
        self.augment_pipe       = augment_pipe
        self.lpips              = lpips
        self.r1_gamma           = r1_gamma
        self.r1_gamma_patch     = r1_gamma_patch
        self.r1_gamma_uv           = r1_gamma_uv
        self.r1_gamma_seg       = r1_gamma_seg
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_init_sigma_patch = blur_init_sigma_patch
        self.blur_fade_kimg     = blur_fade_kimg
        self.blur_patch_seg     = blur_patch_seg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.bg_reg             = True
        self.c_headpose         = False
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        self.patch_scale = patch_scale
        self.masked_sampling = masked_sampling
        self.patch_gan = patch_gan
    

    def run_G(self, shape_params, exp_params, pose_params, eye_pose_params, z, z_bg, c, c_compose, swapping_prob, neural_rendering_resolution, patch_scale=1.0, run_full=True, update_emas=False):

        uv = self.G.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, c, half_size = int(self.G.img_resolution/2))[0]
        head_mask = self.G.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, c, only_face=False,cull_backfaces=False)[1]

        _deformer = self.G._deformer(shape_params,exp_params,pose_params,eye_pose_params)

        c_compose_condition = c_compose.clone()
        if self.c_headpose:
            headpose_rot = self.G.deformer.flame_deform.flame_model._pose2rot(pose_params[...,:3]).reshape(-1,9)
            c_compose_condition = torch.cat([c_compose_condition,headpose_rot],dim=-1)
        
        if self.G.flame_condition:
            c_compose_condition = torch.cat([c_compose_condition,shape_params],dim=-1)

        if swapping_prob is not None:
            c_swapped = torch.roll(c_compose_condition.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c_compose_condition)
        else:
            c_gen_conditioning = torch.zeros_like(c_compose_condition)
        

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, z_bg, c, _deformer, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, smpl_param=(shape_params, exp_params, pose_params, eye_pose_params), patch_scale=patch_scale, run_full=run_full, uv=uv)
        if 'uv' not in gen_output:
            gen_output['uv'] = uv
        if 'head_mask' not in gen_output:
            head_mask = self.G.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, c, only_face=False,cull_backfaces=False)[1]
            gen_output['head_mask'] = head_mask
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                if self.G.has_superresolution:
                    f = torch.arange(-blur_size, blur_size + 1, device=img['image_sr'].device).div(blur_sigma).square().neg().exp2()
                    img['image_sr'] = upfirdn2d.filter2d(img['image_sr'], f / f.sum())
                else:
                    f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                    img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def run_D_patch(self, img, run_patch=True, run_patch_raw=True, update_emas=False, img_name=None):
        patch_raw_logits, patch_logits = None, None
        blur_seg_size = np.floor(self.blur_patch_seg * 3)
        if blur_seg_size > 0:
            f = torch.arange(-blur_seg_size, blur_seg_size + 1, device=img['patch_seg'].device).div(self.blur_patch_seg).square().neg().exp2()
            img['patch_seg'] = upfirdn2d.filter2d(img['patch_seg'], f / f.sum())
        if run_patch:
            if img_name is None:
                img_name = 'patch_image'
            patch_logits = self.D_patch(img, None, img_name=img_name, img_raw_name='patch_image_raw', seg_name='patch_seg', uv_name='patch_uv', update_emas=update_emas)
        if run_patch_raw:
            if img_name is None:
                img_name = 'patch_image_raw'
            patch_raw_logits = self.D_patch(img, None, img_name=img_name, img_raw_name='patch_image_gr', seg_name='patch_seg', uv_name='patch_uv', update_emas=update_emas)
        return patch_logits, patch_raw_logits

    def accumulate_gradients(self, phase, real_img, real_seg, real_uv, real_c, shape_params, exp_params, pose_params, eye_pose_params, gen_z, gen_z_bg, gen_c, gen_c_compose, gain, cur_nimg):
        # assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        if self.r1_gamma_patch == 0:
            phase = {'D_patchreg': 'none', 'D_patchboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma
        r1_gamma_patch = self.r1_gamma_patch
        r1_gamma_uv = self.r1_gamma_uv
        r1_gamma_seg = self.r1_gamma_seg

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        if self.G.has_superresolution:
            real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)
            real_seg_raw = filtered_resizing(real_seg, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

            if self.blur_raw_target:
                blur_size = np.floor(blur_sigma * 3)
                if blur_size > 0:
                    f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                    real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())
                    real_seg_raw = upfirdn2d.filter2d(real_seg_raw, f / f.sum())

            real_img = {'image_sr': real_img, 'uv': real_uv, 'image': real_img_raw, 'seg': torch.mean(real_seg_raw,dim=1,keepdim=True)}
        else:
            real_img = {'image': real_img, 'uv': real_uv, 'seg': torch.mean(real_seg_raw,dim=1,keepdim=True)}
        
        if 'patch' in phase:
            real_img_tmp = real_img['image_sr'].clone()
            real_img_raw_tmp = real_img['image'].clone()
            real_seg_tmp = torch.mean(real_seg,dim=1,keepdim=True).clone()
            real_uv_tmp = real_img['uv'].clone()
            real_img_raw_tmp = torch.nn.functional.interpolate(real_img_raw_tmp, size=(real_img_tmp.shape[-1]),
                                mode='bilinear', align_corners=False, antialias=True)
            real_img_patch, real_img_patch_raw, real_seg_patch, real_uv_patch = [], [], [], []
            if self.masked_sampling:
                eye_mask = self.G.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, real_c, half_size = int(real_img_tmp.shape[-1]/2), eye_mask=True)[1]
                face_wo_eye_mask = self.G.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, real_c, half_size = int(real_img_tmp.shape[-1]/2), face_woeye=True)[1]
                eye_mask = eye_mask * (1-face_wo_eye_mask)
                eye_mask = torch.nn.functional.interpolate(eye_mask, size=(neural_rendering_resolution), mode='bilinear', align_corners=False, antialias=True)
                eye_mask_sr = torch.nn.functional.interpolate(eye_mask, size=(real_img_tmp.shape[-1]), mode='bilinear', align_corners=False, antialias=True)
                head_mask = self.G.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, real_c, half_size = int(real_img_tmp.shape[-1]/2), only_face=False,cull_backfaces=False)[1]
                head_wo_mouth_mask = self.G.deformer.renderer(shape_params, exp_params, pose_params, eye_pose_params, real_c, half_size = int(real_img_tmp.shape[-1]/2), only_face=False,cull_backfaces=True, noinmouth=True)[1]
                mouth_mask = head_mask * (1-head_wo_mouth_mask)
                mouth_mask = torch.nn.functional.interpolate(mouth_mask, size=(neural_rendering_resolution), mode='bilinear', align_corners=False, antialias=True)
                mouth_mask_sr = torch.nn.functional.interpolate(mouth_mask, size=(real_img_tmp.shape[-1]), mode='bilinear', align_corners=False, antialias=True)
                eye_mouth_mask = mouth_mask_sr + eye_mask_sr
                head_mask = torch.nn.functional.interpolate(head_mask, size=(neural_rendering_resolution), mode='bilinear', align_corners=False, antialias=True)
                head_mask_sr = torch.nn.functional.interpolate(head_mask, size=(real_img_tmp.shape[-1]), mode='bilinear', align_corners=False, antialias=True)

            for i in range(real_img_tmp.shape[0]):
                r = torch.rand([])
                mask_guide = None
                if self.masked_sampling>0 and r<self.masked_sampling and eye_mouth_mask[i, 0].sum()>0:
                    mask_guide = eye_mouth_mask[i, 0]
                elif self.masked_sampling>0 and r<self.masked_sampling*2 and head_mask_sr[i,0].sum()>0:
                    mask_guide = head_mask_sr[i, 0]
                if mask_guide is not None:
                    center_idxs = (mask_guide>0).nonzero()
                    idx = torch.randint(len(center_idxs), ()).item()
                    center_idxs = center_idxs[idx]
                    top = center_idxs[0] - neural_rendering_resolution//2
                    left = center_idxs[1] - neural_rendering_resolution//2
                    if top<0 or left<0 or top>real_img_tmp.shape[-1]-neural_rendering_resolution or left>real_img_tmp.shape[-1]-neural_rendering_resolution:
                        top = torch.randint(real_img_tmp.shape[-1]-neural_rendering_resolution+1, ()).item()
                        left = torch.randint(real_img_tmp.shape[-1]-neural_rendering_resolution+1, ()).item()
                else:
                    top = torch.randint(real_img_tmp.shape[-1]-neural_rendering_resolution+1, ()).item()
                    left = torch.randint(real_img_tmp.shape[-1]-neural_rendering_resolution+1, ()).item()
                real_img_patch.append(real_img_tmp[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                real_img_patch_raw.append(real_img_raw_tmp[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                real_seg_patch.append(real_seg_tmp[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                real_uv_patch.append(real_uv_tmp[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
            real_img_patch = torch.cat(real_img_patch, 0)
            real_img_patch_raw = torch.cat(real_img_patch_raw, 0)
            real_seg_patch = torch.cat(real_seg_patch, 0)
            real_uv_patch = torch.cat(real_uv_patch, 0)
            real_img.update({'patch_image': real_img_patch, 'patch_image_raw': real_img_patch_raw, 'patch_seg': real_seg_patch, 'patch_uv': real_uv_patch})

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(shape_params, exp_params, pose_params, eye_pose_params, gen_z, gen_z_bg, gen_c, gen_c_compose, swapping_prob=swapping_prob, 
                    patch_scale=self.patch_scale, run_full=True, neural_rendering_resolution=neural_rendering_resolution)

                c_compose_condition = gen_c_compose.clone()           
                
                gen_logits = self.run_D(gen_img, c_compose_condition, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
                if 'densities_face_ineye' in gen_img:
                    loss_face_eye = (torch.nn.functional.softplus(gen_img['densities_face_ineye']) * gen_img['eye_mask'][:, 0].view(gen_img['eye_mask'].shape[0], -1, 1, 1)).mean(dim=[1,2,3])[:, None] * 10
                    training_stats.report('Loss/G/loss_face_eye', loss_face_eye)
                    loss_eye_density = (lossfun_distortion(gen_img['interval'], gen_img['all_weights'], reduction=None) * gen_img['eye_mask'][:, 0].view(gen_img['eye_mask'].shape[0], -1)).mean(dim=1, keepdim=True)* 100
                    training_stats.report('Loss/G/loss_eye_density', loss_eye_density)
                    loss_eye_T = (gen_img['T_bg'] * gen_img['eye_mask']).mean(dim=[1,2,3])[:, None]* 100
                    training_stats.report('Loss/G/loss_eye_T', loss_eye_T)
                else:
                    loss_face_eye = None
                    loss_eye_T = None
                    loss_eye_density = None
                if 'densities_face_inmouth' in gen_img and gen_img['densities_face_inmouth'] is not None:
                    loss_face_mouth = (torch.nn.functional.softplus(gen_img['densities_face_inmouth']) * gen_img['mouth_mask'][:, 0].view(gen_img['mouth_mask'].shape[0], -1, 1, 1)).mean(dim=[1,2,3])[:, None] * 10
                    training_stats.report('Loss/G/loss_face_mouth', loss_face_mouth)
                    loss_mouth_density = (lossfun_distortion(gen_img['interval'], gen_img['all_weights'], reduction=None) * gen_img['mouth_mask'][:, 0].view(gen_img['mouth_mask'].shape[0], -1)).mean(dim=1, keepdim=True)* 100
                    training_stats.report('Loss/G/loss_mouth_density', loss_mouth_density)
                    loss_mouth_T = (gen_img['T_bg'] * gen_img['mouth_mask']).mean(dim=[1,2,3])[:, None]* 100
                    training_stats.report('Loss/G/loss_mouth_T', loss_mouth_T)
                else:
                    loss_face_mouth = None
                    loss_mouth_density = None
                    loss_mouth_T = None
                    
                
                loss_Glpips = loss_Gpatch = loss_patch_mouth_density = loss_patch_mouth_T = loss_patch_eye_density = loss_patch_eye_T = None
                if 'patch_image_raw' in gen_img:
                    loss_Glpips = self.lpips(gen_img['patch_image_raw'], gen_img['patch_image'])
                    training_stats.report('Loss/G/lpips', loss_Glpips)
                    _, patch_raw_logits = self.run_D_patch(gen_img, run_patch=False)
                    training_stats.report('Loss/scores/patch_fake', patch_raw_logits)
                    training_stats.report('Loss/signs/patch_fake', patch_raw_logits.sign())
                    loss_Gpatch = torch.nn.functional.softplus(-patch_raw_logits) * self.patch_gan
                    training_stats.report('Loss/G/patch', loss_Gpatch)
                    loss_patch_mouth_density = (lossfun_distortion(gen_img['patch_all_depths'], gen_img['patch_all_weights'], reduction=None) * gen_img['patch_mouth_mask'][:, 0].view(gen_img['patch_mouth_mask'].shape[0], -1)).mean(dim=1, keepdim=True)* 100
                    training_stats.report('Loss/G/patch_mouth_density', loss_mouth_density)
                    loss_patch_mouth_T = (gen_img['patch_T_bg'] * gen_img['patch_mouth_mask']).mean(dim=[1,2,3])[:, None]* 100
                    training_stats.report('Loss/G/loss_patch_mouth_T', loss_patch_mouth_T)
                    loss_patch_eye_density = (lossfun_distortion(gen_img['patch_all_depths'], gen_img['patch_all_weights'], reduction=None) * gen_img['patch_eye_mask'][:, 0].view(gen_img['patch_eye_mask'].shape[0], -1)).mean(dim=1, keepdim=True)* 100
                    training_stats.report('Loss/G/patch_eye_density', loss_eye_density)
                    loss_patch_eye_T = (gen_img['patch_T_bg'] * gen_img['patch_eye_mask']).mean(dim=[1,2,3])[:, None]* 100
                    training_stats.report('Loss/G/loss_patch_eye_T', loss_patch_eye_T)
                    
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_G = loss_Gmain.mean()
                if loss_face_eye is not None:
                    loss_G += loss_face_eye.mean()
                if loss_face_mouth is not None:
                    loss_G += loss_face_mouth.mean()
                if loss_Glpips is not None:
                    loss_G += loss_Glpips.mean()
                if loss_Gpatch is not None:
                    loss_G += loss_Gpatch.mean()
                if loss_mouth_density is not None:
                    loss_G += loss_mouth_density.mean()    
                if loss_patch_mouth_density is not None:
                    loss_G += loss_patch_mouth_density.mean()
                if loss_mouth_T is not None:
                    loss_G += loss_mouth_T.mean()    
                if loss_patch_mouth_T is not None:
                    loss_G += loss_patch_mouth_T.mean()
                if loss_eye_density is not None:
                    loss_G += loss_eye_density.mean()    
                if loss_patch_eye_density is not None:
                    loss_G += loss_patch_eye_density.mean()
                if loss_eye_T is not None:
                    loss_G += loss_eye_T.mean()    
                if loss_patch_eye_T is not None:
                    loss_G += loss_patch_eye_T.mean()
                loss_G.mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':

            c_compose_condition = gen_c_compose.clone()
            if self.c_headpose:
                headpose_rot = self.G.deformer.flame_deform.flame_model._pose2rot(pose_params[...,:3]).reshape(-1,9)
                c_compose_condition = torch.cat([c_compose_condition,headpose_rot],dim=-1)
    
            if self.G.flame_condition:
                c_compose_condition = torch.cat([c_compose_condition,shape_params],dim=-1)

            if swapping_prob is not None:
                c_swapped = torch.roll(c_compose_condition.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c_compose.device) < swapping_prob, c_swapped, c_compose_condition)
            else:
                c_gen_conditioning = torch.zeros_like(c_compose_condition)
            
            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            # initial_coordinates = initial_coordinates*self.G.rendering_kwargs['box_warp']*0.5 # add to ensure correct scale
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist'] * self.G.rendering_kwargs['box_warp']

            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            out = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), shape_params,exp_params,pose_params,eye_pose_params, ws, update_emas=False)
            if isinstance(out, tuple):
                TVloss = 0
                for out_ in out:
                    sigma = out_['sigma'][:,:initial_coordinates.shape[1]*2]
                    sigma_initial = sigma[:, :sigma.shape[1]//2]
                    sigma_perturbed = sigma[:, sigma.shape[1]//2:]
                    TVloss += torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg'] / len(out)
                training_stats.report('Loss/G/TVloss', TVloss)
            else:
                sigma = out['sigma'][:,:initial_coordinates.shape[1]*2]
                sigma_initial = sigma[:, :sigma.shape[1]//2]
                sigma_perturbed = sigma[:, sigma.shape[1]//2:]

                TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
                training_stats.report('Loss/G/TVloss', TVloss)
            
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(shape_params, exp_params, pose_params, eye_pose_params, gen_z, gen_z_bg, gen_c, gen_c_compose, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)

                c_compose_condition = gen_c_compose.clone()                
                
                gen_logits = self.run_D(gen_img, c_compose_condition, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                if self.G.has_superresolution:
                    real_img_tmp_image = real_img['image_sr'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_uv_tmp_image = real_img['uv'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_img_tmp_image_raw = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_seg_tmp_image = real_img['seg'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_img_tmp = {'image_sr': real_img_tmp_image, 'image': real_img_tmp_image_raw, 'uv': real_uv_tmp_image, 'seg':real_seg_tmp_image}
                else:
                    real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_uv_tmp_image = real_img['uv'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_seg_tmp_image = real_img['seg'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_img_tmp = {'image': real_img_tmp_image, 'uv': real_uv_tmp_image, 'seg':real_seg_tmp_image}

                c_compose_condition = real_c.clone()

                real_logits = self.run_D(real_img_tmp, c_compose_condition, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            if self.G.has_superresolution:
                                if self.D.has_uv and self.D.has_seg:
                                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image_sr'],real_img_tmp['image'], real_img_tmp['uv'], real_img_tmp['seg']], create_graph=True, only_inputs=True)
                                    r1_grads_uv = r1_grads[2]
                                    r1_grads_seg = r1_grads[3]
                                elif self.D.has_uv:
                                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image_sr'],real_img_tmp['image'], real_img_tmp['uv']], create_graph=True, only_inputs=True)
                                    r1_grads_uv = r1_grads[2]
                                    r1_grads_seg = torch.zeros_like(r1_grads_uv)
                                else:
                                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image_sr'],real_img_tmp['image']], create_graph=True, only_inputs=True)
                                    r1_grads_uv = torch.zeros_like(r1_grads[0])
                                    r1_grads_seg = torch.zeros_like(r1_grads[0])
                                r1_grads_image = r1_grads[0]
                                r1_grads_image_raw = r1_grads[1]
                                
                            else:
                                if self.D.has_uv:
                                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['uv']], create_graph=True, only_inputs=True)
                                    r1_grads_uv = r1_grads[1]
                                else:
                                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                                    r1_grads_uv = torch.zeros_like(r1_grads[0])
                                r1_grads_image = r1_grads[0]
                                r1_grads_image_raw = torch.zeros_like(r1_grads_image)
                                
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                        r1_penalty_uv =  r1_grads_uv.square().sum([1,2,3])
                        r1_penalty_seg = r1_grads_seg.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            if self.G.has_superresolution:
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image_sr']], create_graph=True, only_inputs=True)
                            else:
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                        r1_penalty_uv = torch.zeros_like(r1_penalty)
                    loss_Dr1 = r1_penalty * (r1_gamma / 2) + r1_penalty_uv * (r1_gamma_uv / 2) + r1_penalty_seg * (r1_gamma_seg / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        if phase in ['D_patchmain', 'D_patchboth']:
            with torch.autograd.profiler.record_function('D_patchmain_forward'):
                gen_img, _gen_ws = self.run_G(shape_params, exp_params, pose_params, eye_pose_params, gen_z, gen_z_bg, gen_c, gen_c_compose, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, patch_scale=self.patch_scale, run_full=False, update_emas=True)
                _, patch_raw_logits = self.run_D_patch(gen_img, run_patch=False)
                patch_logits, _ = self.run_D_patch(real_img, run_patch_raw=False)
                training_stats.report('Loss/scores/patch_fake', patch_raw_logits)
                training_stats.report('Loss/signs/patch_fake', patch_raw_logits.sign())
                training_stats.report('Loss/scores/patch_real', patch_logits)
                training_stats.report('Loss/signs/patch_real', patch_logits.sign())
                loss_D_patch_raw = torch.nn.functional.softplus(patch_raw_logits)
                loss_D_patch = torch.nn.functional.softplus(-patch_logits)
                training_stats.report('Loss/D_patch/loss', loss_D_patch_raw+loss_D_patch)

            with torch.autograd.profiler.record_function('D_patchmain_backward'):
                (loss_D_patch_raw+loss_D_patch).mean().mul(gain).backward()
        
        if phase in ['D_patchreg', 'D_patchboth']:
            with torch.autograd.profiler.record_function('D_patchreg_forward'):
                real_img['patch_image'].requires_grad_(True)
                patch_logits, _ = self.run_D_patch(real_img, run_patch_raw=False)
                r1_grads = torch.autograd.grad(outputs=[patch_logits.sum()], inputs=[real_img['patch_image']], create_graph=True, only_inputs=True)
                r1_grads_image = r1_grads[0]
                r1_penalty = r1_grads_image.square().sum([1,2,3])
                loss_Dpatch_r1 = r1_penalty * (r1_gamma_patch / 2)
                training_stats.report('Loss/patch_r1_penalty', r1_penalty)
                training_stats.report('Loss/D_patch/reg', loss_Dpatch_r1)

            with torch.autograd.profiler.record_function('D_patchreg_backward'):
                loss_Dpatch_r1.mean().mul(gain).backward()