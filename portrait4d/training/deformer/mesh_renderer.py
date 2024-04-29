# modified from Deep3DFaceRecon_pytorch: https://github.com/sicxu/Deep3DFaceRecon_pytorch
# use pytorch3d as renderer instead of nvdiffrast as in Deep3DFaceRecon_pytorch

import torch
import torch.nn.functional as F
# import kornia
# from kornia.geometry.camera import pixel2cam
import numpy as np
from typing import List
# import nvdiffrast.torch as dr
from scipy.io import loadmat
from torch import nn

import pytorch3d.ops
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)

# def ndc_projection(x=0.1, n=1.0, f=50.0):
#     return np.array([[n/x,    0,            0,              0],
#                      [  0, n/-x,            0,              0],
#                      [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
#                      [  0,    0,           -1,              0]]).astype(np.float32)

class MeshRenderer(nn.Module):
    def __init__(self,
                znear=0.1,
                zfar=10):
        super(MeshRenderer, self).__init__()

        self.znear = znear
        self.zfar = zfar

        self.rasterizer = MeshRasterizer()
    
    def forward(self, fov, rasterize_size, vertex, tri, feat=None, cull_backfaces=True, perspective=True, scale=1.0):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3), z axis points inward
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, C), features
        """
        device = vertex.device
        rsize = int(rasterize_size)
        C = feat.shape[-1]

        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 0] = -vertex[..., 0]

        tri = tri.type(torch.int32).contiguous()

        if perspective:
            cameras = FoVPerspectiveCameras(
                device=device,
                fov=fov,
                znear=self.znear,
                zfar=self.zfar,
            )
        else:
            cameras = FoVOrthographicCameras(
                device=device,
                scale_xyz=((scale, scale, scale),),
                znear=self.znear,
                zfar=self.zfar,
            )

        raster_settings = RasterizationSettings(
            cull_backfaces=cull_backfaces,
            image_size=rsize
        )

        mesh = Meshes(vertex[...,:3], tri)

        fragments = self.rasterizer(mesh, cameras = cameras, raster_settings = raster_settings)
        rast_out = fragments.pix_to_face.squeeze(-1)
        depth = fragments.zbuf

        # render depth
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out >= 0).float().unsqueeze(1)
        depth = mask * depth
        
        image = None
        if feat is not None:
            attributes = feat.reshape(-1,C)[mesh.faces_packed()]
            image = pytorch3d.ops.interpolate_face_attributes(fragments.pix_to_face,
                                                      fragments.bary_coords,
                                                      attributes)
            image = image.squeeze(-2).permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image

