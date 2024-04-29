# Modified from EG3D: https://github.com/NVlabs/eg3d

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn
from torch_utils import persistence

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """

    N, M, _ = coordinates.shape
    xy_coords = coordinates[..., [0, 1]]
    yz_coords = coordinates[..., [1, 2]]
    zx_coords = coordinates[..., [2, 0]]
    
    return torch.stack([xy_coords, yz_coords, zx_coords], dim=1).reshape(N*3, M, 2)

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None, box_warp_scale=1.0):
    assert padding_mode == 'zeros'
    # N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape

    coordinates = (2/(box_warp_scale*box_warp)) * coordinates # TODO: add specific box bounds
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    if isinstance(plane_features, list):
        for i, p in enumerate(plane_features):
            N, n_planes, C, H, W = p.shape
            p = p.reshape(N*n_planes, C, H, W)
            if i==0:
                output_features = torch.nn.functional.grid_sample(p, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
            else:
                output_features += torch.nn.functional.grid_sample(p, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    else:
        N, n_planes, C, H, W = plane_features.shape
        plane_features = plane_features.reshape(N*n_planes, C, H, W)
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        return rgb_final, depth_final, weights.sum(2)

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options, dynamic_mask=None, box_warp_scale=1.0):

        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'], box_warp_scale=box_warp_scale)

        if decoder is None:
            return {'triplane_feature': sampled_features.mean(1)}
        else:
            out = decoder(sampled_features, sample_directions)
            out['triplane_feature'] = sampled_features.mean(1)
            if options.get('density_noise', 0) > 0:
                out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
            return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

#----------------------------------------------------------------------------------------------
# Renderer for a naive generator
@persistence.persistent_class
class DeformImportanceRenderer(ImportanceRenderer):
    def __init__(self):
        super().__init__()
    
    def forward(self, planes, decoder, deformer, ray_origins, ray_directions, rendering_options, dynamic=False, cam_z=None):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        elif rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto-plane':
            # cam_z: [batchsize], depth(z) of world space origin in the camera space
            cam_z = cam_z.unsqueeze(-1).unsqueeze(-1)
            ray_start = torch.abs(cam_z) - rendering_options['box_warp']/2
            ray_end = torch.abs(cam_z) + rendering_options['box_warp']/2
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
            depths_coarse = depths_coarse.repeat(1,ray_origins.shape[1],1,1)        
        else:
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        # Add deformation from target space to canonical space
        out_deform = deformer(sample_coordinates)
        sample_coordinates_ca = out_deform['canonical']
        coarse_offset = out_deform['offset']
        coarse_dist = out_deform['dist_to_surface'].reshape(batch_size,num_rays,-1)

        out = self.run_model(planes, decoder, sample_coordinates_ca, sample_directions, rendering_options, dynamic_mask=out_deform['dynamic_mask'])
        triplane_features_coarse = out['triplane_feature']
        sample_coordinates_coarse = sample_coordinates        
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights, _ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            
            # Add deformation from target space to canonical space
            out_deform = deformer(sample_coordinates)
            sample_coordinates_ca = out_deform['canonical']
            fine_offset = out_deform['offset']
            fine_dist = out_deform['dist_to_surface'].reshape(batch_size,num_rays,-1)

            out = self.run_model(planes, decoder, sample_coordinates_ca, sample_directions, rendering_options, dynamic_mask=out_deform['dynamic_mask'])
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights, T_bg = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights, T_bg = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)


        return rgb_final, depth_final, all_depths, weights, T_bg, torch.cat([coarse_offset,fine_offset],dim=1), torch.cat([coarse_dist,fine_dist],dim=-1),out_deform['vts_mask'], out_deform['vts_mask_region'], sample_coordinates_coarse, triplane_features_coarse

#----------------------------------------------------------------------------------------------
# Renderer for GenHead
class PartDeformImportanceRenderer(ImportanceRenderer): # 192
    def __init__(self):
        super().__init__()
    
    def forward(self, planes, decoder, deformer, ray_origins, ray_directions, rendering_options, eye_mask=None, mouth_mask=None, mouth_dynamic=False, auto_fuse=True, cam_z=None):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        elif rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto-plane':
            # cam_z: [batchsize], depth(z) of world space origin in the camera space
            cam_z = cam_z.unsqueeze(-1).unsqueeze(-1)
            ray_start = torch.abs(cam_z) - rendering_options['box_warp']/2
            ray_end = torch.abs(cam_z) + rendering_options['box_warp']/2
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
            depths_coarse = depths_coarse.repeat(1,ray_origins.shape[1],1,1)        
        else:
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        # Add deformation from target space to canonical space
        out_deform = deformer(sample_coordinates, mouth=mouth_mask is not None)
        sample_coordinates_ca_eye = out_deform['canonical_eye']
        sample_coordinates_ca_face = out_deform['canonical_face']
        coarse_dist_eye = out_deform['dist_to_eye_surface'].reshape(batch_size, num_rays, samples_per_ray)
        coarse_dist_face = out_deform['dist_to_face_surface'].reshape(batch_size, num_rays, samples_per_ray)
        coarse_inside_bbox_eye = out_deform['inside_bbox_eye'].reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # mouth_mask[:] = 1
        # eye_mask[:] = 0
        if mouth_mask is not None:
            sample_coordinates_ca_mouth = out_deform['canonical_mouth']
            coarse_dist_mouth = out_deform['dist_to_mouth_surface'].reshape(batch_size, num_rays, samples_per_ray)
            mouth_mask_sel = mouth_mask.repeat(1,1,coarse_dist_mouth.shape[-1])
            mouth_mask_sel_pts = mouth_mask_sel.view(mouth_mask_sel.shape[0], -1, 1).repeat(1,1,3)
            sample_coordinates_ca_eye = torch.where(mouth_mask_sel_pts>0, sample_coordinates_ca_mouth, sample_coordinates_ca_eye)
        if out_deform['dynamic_mask'] is not None: # Deprecated
            dynamic_mask = out_deform['dynamic_mask'] * (1-coarse_inside_bbox_eye.view(coarse_inside_bbox_eye.shape[0], -1, 1))
            if mouth_dynamic:
                plane_eye = [torch.cat([p[:, :3],p[:, -3:]],dim=2) for p in planes]
            else:
                plane_eye = [p[:, :3] for p in planes]
            plane_face = [torch.cat([p[:, 3:6],p[:, -3:]],dim=2) for p in planes]
        else:
            dynamic_mask = None
            plane_eye = [p[:, :3] for p in planes]
            plane_face = [p[:, 3:6] for p in planes]

        out_eye = self.run_model(plane_eye, decoder, sample_coordinates_ca_eye, sample_directions, rendering_options, dynamic_mask=(None, dynamic_mask)[mouth_dynamic]) # do not use dynamic textures
        colors_coarse_eye = out_eye['rgb'].reshape(batch_size, num_rays, samples_per_ray, out_eye['rgb'].shape[-1])
        densities_coarse_eye = out_eye['sigma'].reshape(batch_size, num_rays, samples_per_ray, 1)
        if mouth_mask is not None:
            densities_coarse_eye = densities_coarse_eye * (coarse_inside_bbox_eye+mouth_mask_sel[..., None])
        else:
            densities_coarse_eye = densities_coarse_eye * coarse_inside_bbox_eye
            
        out_face = self.run_model(plane_face, decoder, sample_coordinates_ca_face, sample_directions, rendering_options, dynamic_mask=dynamic_mask)
        triplane_features_coarse_face = out_face['triplane_feature']
        sample_coordinates_coarse = sample_coordinates
        colors_coarse_face = out_face['rgb'].reshape(batch_size, num_rays, samples_per_ray, out_face['rgb'].shape[-1])
        densities_coarse_face = out_face['sigma'].reshape(batch_size, num_rays, samples_per_ray, 1)
        densities_coarse_face_ineye = densities_coarse_face * coarse_inside_bbox_eye

        eye_mask_sel = eye_mask.repeat(1,1,densities_coarse_face.shape[-2])
        eye_mask_sel_coarse = eye_mask_sel * coarse_inside_bbox_eye[..., 0]
        if mouth_mask is not None:
            densities_coarse_face_inmouth = densities_coarse_face * mouth_mask_sel[..., None]

        # eye_mask_sel_coarse[:] = 0.
        # mouth_mask_sel[:] = 1.
        if mouth_mask is not None:
            if not auto_fuse:
                densities_coarse = densities_coarse_eye * eye_mask_sel_coarse[..., None] + densities_coarse_eye * mouth_mask_sel[..., None] + densities_coarse_face * (1-eye_mask_sel_coarse[..., None]-mouth_mask_sel[..., None])
                colors_coarse = colors_coarse_eye * eye_mask_sel_coarse[..., None] + colors_coarse_eye * mouth_mask_sel[..., None] + colors_coarse_face * (1-eye_mask_sel_coarse[..., None]-mouth_mask_sel[..., None])
            else:
                densities_coarse_1 = densities_coarse_eye * eye_mask_sel_coarse[..., None] + densities_coarse_eye * mouth_mask_sel[..., None] + densities_coarse_face * (1-eye_mask_sel_coarse[..., None]-mouth_mask_sel[..., None])
                colors_coarse_1 = colors_coarse_eye * eye_mask_sel_coarse[..., None] + colors_coarse_eye * mouth_mask_sel[..., None] + colors_coarse_face * (1-eye_mask_sel_coarse[..., None]-mouth_mask_sel[..., None])
                densities_coarse_2 = densities_coarse_eye + densities_coarse_face
                density_w = torch.cat([densities_coarse_eye[..., None], densities_coarse_face[..., None]], dim=-1)
                density_w = torch.softmax(density_w, dim=-1)
                colors_coarse_2 = density_w[..., 0] * colors_coarse_eye + density_w[..., 1] * colors_coarse_face
                m = (mouth_mask_sel>0).float() #mouth_mask_hard_sel if mouth_mask_hard_sel is not None else mouth_mask_sel
                densities_coarse = densities_coarse_1 * (1-m[..., None]) + densities_coarse_2 * (m[..., None])
                colors_coarse = colors_coarse_1 * (1-m[..., None]) + colors_coarse_2 * (m[..., None])
            coarse_dist = coarse_dist_eye * eye_mask_sel_coarse + coarse_dist_mouth * mouth_mask_sel  + coarse_dist_face * (1-eye_mask_sel_coarse-mouth_mask_sel)
        else:
            densities_coarse = densities_coarse_eye * eye_mask_sel_coarse[..., None] + densities_coarse_face * (1-eye_mask_sel_coarse[..., None])
            colors_coarse = colors_coarse_eye * eye_mask_sel_coarse[..., None] + colors_coarse_face * (1-eye_mask_sel_coarse[..., None])
            coarse_dist = coarse_dist_eye * eye_mask_sel_coarse + coarse_dist_face * (1-eye_mask_sel_coarse)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights, _ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            # sample_coordinates_ca = deformer(sample_coordinates)
            out_deform = deformer(sample_coordinates, mouth=mouth_mask is not None)
            sample_coordinates_ca_eye = out_deform['canonical_eye']
            sample_coordinates_ca_face = out_deform['canonical_face']
            fine_dist_eye = out_deform['dist_to_eye_surface'].reshape(batch_size, num_rays, samples_per_ray)
            fine_dist_face = out_deform['dist_to_face_surface'].reshape(batch_size, num_rays, samples_per_ray)
            fine_inside_bbox_eye = out_deform['inside_bbox_eye'].reshape(batch_size, num_rays, samples_per_ray, 1)
            if mouth_mask is not None:
                sample_coordinates_ca_mouth = out_deform['canonical_mouth']
                fine_dist_mouth = out_deform['dist_to_mouth_surface'].reshape(batch_size, num_rays, samples_per_ray)
                sample_coordinates_ca_eye = torch.where(mouth_mask_sel_pts>0, sample_coordinates_ca_mouth, sample_coordinates_ca_eye)
            if out_deform['dynamic_mask'] is not None: # Deprecated
                dynamic_mask = out_deform['dynamic_mask'] * (1-fine_inside_bbox_eye.view(fine_inside_bbox_eye.shape[0], -1, 1))
            else:
                dynamic_mask = None

            out_eye = self.run_model(plane_eye, decoder, sample_coordinates_ca_eye, sample_directions, rendering_options, dynamic_mask=(None, dynamic_mask)[mouth_dynamic])
            colors_fine_eye = out_eye['rgb'].reshape(batch_size, num_rays, samples_per_ray, out_eye['rgb'].shape[-1])
            densities_fine_eye = out_eye['sigma'].reshape(batch_size, num_rays, samples_per_ray, 1)
            if mouth_mask is not None:
                densities_fine_eye = densities_fine_eye * (fine_inside_bbox_eye + mouth_mask_sel[..., None])
            else:
                densities_fine_eye = densities_fine_eye * fine_inside_bbox_eye

            out_face = self.run_model(plane_face, decoder, sample_coordinates_ca_face, sample_directions, rendering_options, dynamic_mask=dynamic_mask)
            colors_fine_face = out_face['rgb'].reshape(batch_size, num_rays, samples_per_ray, out_face['rgb'].shape[-1])
            densities_fine_face = out_face['sigma'].reshape(batch_size, num_rays, samples_per_ray, 1)
            densities_fine_face_ineye = densities_fine_face * fine_inside_bbox_eye
            if mouth_mask is not None:
                densities_fine_face_inmouth = densities_fine_face * mouth_mask_sel[..., None]

            eye_mask_sel_fine = eye_mask_sel * fine_inside_bbox_eye[..., 0]
            # eye_mask_sel_fine[:] = 0.
            if mouth_mask is not None:
                if not auto_fuse:
                    densities_fine = densities_fine_eye * eye_mask_sel_fine[..., None] + densities_fine_eye * mouth_mask_sel[..., None] + densities_fine_face * (1-eye_mask_sel_fine[..., None]-mouth_mask_sel[..., None])
                    colors_fine = colors_fine_eye * eye_mask_sel_fine[..., None] + colors_fine_eye * mouth_mask_sel[..., None] + colors_fine_face * (1-eye_mask_sel_fine[..., None]-mouth_mask_sel[..., None])
                else:
                    densities_fine_1 = densities_fine_eye * eye_mask_sel_fine[..., None] + densities_fine_eye * mouth_mask_sel[..., None] + densities_fine_face * (1-eye_mask_sel_fine[..., None]-mouth_mask_sel[..., None])
                    colors_fine_1 = colors_fine_eye * eye_mask_sel_fine[..., None] + colors_fine_eye * mouth_mask_sel[..., None] + colors_fine_face * (1-eye_mask_sel_fine[..., None]-mouth_mask_sel[..., None])
                    densities_fine_2 = densities_fine_eye + densities_fine_face
                    density_w = torch.cat([densities_fine_eye[..., None], densities_fine_face[..., None]], dim=-1)
                    density_w = torch.softmax(density_w, dim=-1)
                    colors_fine_2 = density_w[..., 0] * colors_fine_eye + density_w[..., 1] * colors_fine_face
                    densities_fine = densities_fine_1 * (1-m[..., None]) + densities_fine_2 * (m[..., None])
                    colors_fine = colors_fine_1 * (1-m[..., None]) + colors_fine_2 * (m[..., None])
                fine_dist = fine_dist_eye * eye_mask_sel_fine + fine_dist_mouth * mouth_mask_sel + fine_dist_face * (1-eye_mask_sel_fine-mouth_mask_sel)
            else:
                densities_fine = densities_fine_eye * eye_mask_sel_fine[..., None] + densities_fine_face * (1-eye_mask_sel_fine[..., None])
                colors_fine = colors_fine_eye * eye_mask_sel_fine[..., None] + colors_fine_face * (1-eye_mask_sel_fine[..., None])
                fine_dist = fine_dist_eye * eye_mask_sel_fine + fine_dist_face * (1-eye_mask_sel_fine)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)
            face_densities_ineye = torch.cat([densities_coarse_face_ineye, densities_fine_face_ineye], dim = -2)
            
            # Aggregate
            rgb_final, depth_final, weights, T_bg = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
            if mouth_mask is not None:
                face_densities_inmouth = torch.cat([densities_coarse_face_inmouth, densities_fine_face_inmouth], dim = -2)
            else:
                face_densities_inmouth = None
        else:
            rgb_final, depth_final, weights, T_bg = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        return rgb_final, depth_final, all_depths, weights, T_bg, torch.zeros_like(sample_coordinates_ca_eye), torch.cat([coarse_dist,fine_dist],dim=-1), face_densities_ineye, face_densities_inmouth, out_deform['vts_mask'], out_deform['vts_mask_region'], sample_coordinates_coarse, triplane_features_coarse_face, eye_mask_sel_coarse, mouth_mask_sel

#----------------------------------------------------------------------------------------------
# Renderer for Portrait4D
@persistence.persistent_class
class DeformImportanceRendererNew(ImportanceRenderer):
    def __init__(self):
        super().__init__()
    
    def forward(self, planes, decoder, deformer, ray_origins, ray_directions, rendering_options, cam_z=None, box_warp_scale=1.0):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        elif rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto-plane':
            # cam_z: [batchsize], depth(z) of world space origin in the camera space
            cam_z = cam_z.unsqueeze(-1).unsqueeze(-1)
            ray_start = torch.abs(cam_z) - box_warp_scale*rendering_options['box_warp']/2
            ray_end = torch.abs(cam_z) + box_warp_scale*rendering_options['box_warp']/2
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
            depths_coarse = depths_coarse.repeat(1,ray_origins.shape[1],1,1)
        else:
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        # Add deformation from target space to canonical space
        out_deform = deformer(sample_coordinates)
        sample_coordinates_ca = out_deform['canonical']

        out = self.run_model(planes, decoder, sample_coordinates_ca, sample_directions, rendering_options, box_warp_scale=box_warp_scale)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights, _ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            
            # Add deformation from target space to canonical space
            out_deform = deformer(sample_coordinates)
            sample_coordinates_ca = out_deform['canonical']

            out = self.run_model(planes, decoder, sample_coordinates_ca, sample_directions, rendering_options, box_warp_scale=box_warp_scale)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights, T_bg = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights, T_bg = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)


        return rgb_final, depth_final, all_depths, weights, T_bg