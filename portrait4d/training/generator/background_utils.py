# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# import sys
# sys.path.append('../../code_v1.0')

import torch
import numpy as np
from torch_utils import persistence
from models.stylegan.networks_stylegan2 import MappingNetwork, SynthesisLayer
from torch_utils import misc
from training.volumetric_rendering.ray_sampler import RaySampler
from training.volumetric_rendering.math_utils import normalize_vecs
import dnnlib

# def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
#     """
#     Normalize vector lengths.
#     """
#     return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

# class RaySampler(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None


#     def forward(self, cam2world_matrix, intrinsics, resolution):
#         """
#         Create batches of rays and return origins and directions.

#         cam2world_matrix: (N, 4, 4)
#         intrinsics: (N, 3, 3)
#         resolution: int

#         ray_origins: (N, M, 3)
#         ray_dirs: (N, M, 2)
#         """
#         N, M = cam2world_matrix.shape[0], resolution**2
#         cam_locs_world = cam2world_matrix[:, :3, 3]
#         fx = intrinsics[:, 0, 0]
#         fy = intrinsics[:, 1, 1]
#         cx = intrinsics[:, 0, 2]
#         cy = intrinsics[:, 1, 2]
#         sk = intrinsics[:, 0, 1]

#         # uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), indexing='ij')) * (1./resolution) + (0.5/resolution)
#         uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), indexing='ij')) + 0.5
#         uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
#         uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

#         x_cam = uv[:, :, 0].view(N, -1)
#         y_cam = uv[:, :, 1].view(N, -1)
#         # z_cam = torch.ones((N, M), device=cam2world_matrix.device) # Original EG3D implementation, z points inward
#         z_cam = - torch.ones((N, M), device=cam2world_matrix.device) # Our camera space coordinate

#         x_lift = - (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
#         y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

#         cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

#         world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

#         ray_dirs = world_rel_points - cam_locs_world[:, None, :]
#         ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

#         ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

#         return ray_origins, ray_dirs

def get_dist_from_origin(ray_origins,ray_dirs):
    ray_dirs = normalize_vecs(ray_dirs)
    dist = torch.sqrt(torch.sum(ray_origins**2,dim=-1)-torch.sum(ray_origins*ray_dirs,dim=-1)**2 + 1e-10)

    return dist

@torch.no_grad()
def get_intersections_with_sphere(ray_origins,ray_dirs,radius):
    dist = get_dist_from_origin(ray_origins,ray_dirs)
    valid_mask = (dist <= radius)

    intersections = torch.zeros_like(ray_dirs)
    intersections[valid_mask] = ray_origins[valid_mask] + (-torch.sum(ray_origins[valid_mask]*ray_dirs[valid_mask],dim=-1,keepdim=True) + \
    torch.sqrt(radius**2 + torch.sum(ray_origins[valid_mask]*ray_dirs[valid_mask],dim=-1,keepdim=True)**2 - torch.sum(ray_origins[valid_mask]**2,dim=-1,keepdim=True)))*ray_dirs[valid_mask]

    return intersections, valid_mask

@torch.no_grad()
def get_theta_phi_bg(ray_origins,ray_dirs,radius):
    intersections, valid_mask = get_intersections_with_sphere(ray_origins,ray_dirs,radius)
    phi = torch.zeros_like(intersections[...,0])
    theta = torch.zeros_like(intersections[...,0])

    phi[valid_mask] = torch.arcsin(torch.clamp(intersections[valid_mask][...,1]/radius,-1+1e-10,1-1e-10))

    radius_xz = torch.sqrt(intersections[valid_mask][...,0]**2+intersections[valid_mask][...,2]**2)
    theta_no_sign = torch.arccos(torch.clamp(torch.div(intersections[valid_mask][...,2],radius_xz),-1+1e-10,1-1e-10))
    theta[valid_mask] = torch.where(intersections[valid_mask][...,0]>torch.zeros_like(intersections[valid_mask][...,0]),theta_no_sign,2*np.pi-theta_no_sign)

    # normalizing to [-1,1]
    # theta = (theta-np.pi)/np.pi # times 2 because the theta can hardly exceed pi for frontal-facing scene
    theta = torch.sin(theta)
    phi = torch.sin(phi)

    return torch.stack([theta,phi],dim=-1), valid_mask

@persistence.persistent_class
class BGSynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_channels,               # Number of color channels.
        hidden_channels = 64,
        L               = 10,
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.L = L

        self.num_ws = 0
        for idx in range(5):
            in_channels = L*4 if idx == 0 else hidden_channels
            out_channels = hidden_channels if idx < 4 else img_channels
            activation = 'lrelu' if idx < 4 else 'sigmoid'
            layer = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=64, kernel_size=1, use_noise=False, activation=activation, **block_kwargs)
            self.num_ws += 1
            setattr(self, f'b{idx}', layer)
        
    def positional_encoding(self, p, use_pos=False):

        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * np.pi * p),
            torch.cos((2 ** i) * np.pi * p)],
            dim=-1) for i in range(self.L)], dim=-1)
        if use_pos:
            p_transformed = torch.cat([p_transformed, p], -1)
        return p_transformed

    def forward(self, ws, x, update_emas, **block_kwargs):
        _ = update_emas
        layer_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for idx in range(5):
                layer = getattr(self, f'b{idx}')
                layer_ws.append(ws[:,w_idx])
                w_idx += 1

        x = self.positional_encoding(x) # (N,M,L*4)
        x = x.permute(0,2,1).unsqueeze(-1) # (N,L*4,M,1)

        for idx, cur_ws in enumerate(layer_ws):
            layer = getattr(self, f'b{idx}')
            x = layer(x, cur_ws, **block_kwargs)

        return x

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'hidden_channels={self.hidden_channels:d}, img_channels={self.img_channels:d},',
            f'L={self.L:d}'])


@persistence.persistent_class
class BGGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_channels = img_channels
        self.synthesis = BGSynthesisNetwork(w_dim=w_dim, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, x, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, x, update_emas=update_emas, **synthesis_kwargs)
        return img


if __name__=='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from PIL import Image

    ray_sampler = RaySampler()

    camera_params = torch.eye(4).unsqueeze(0)
    camera_params[...,2,3] += 4.5
    intrinsics = torch.eye(3).unsqueeze(0)
    intrinsics[:,0,0] = 300
    intrinsics[:,1,1] = 300
    intrinsics[:,0,2] = 32
    intrinsics[:,1,2] = 32

    neural_rendering_resolution = 64

    ray_origins, ray_directions = ray_sampler(camera_params, intrinsics, neural_rendering_resolution)

    # print(ray_directions)

    angles, valid_mask = get_theta_phi_bg(ray_origins,ray_directions,radius=1.0)

    angles = angles.reshape(1,64,64,2)
    
    print(angles[0,31])
    print(angles.shape)
    print(valid_mask.shape)

    valid_mask = valid_mask.reshape(1,64,64).squeeze(0).numpy().astype(np.uint8)*255

    Image.fromarray(valid_mask, 'L').save('bg_mask.png')