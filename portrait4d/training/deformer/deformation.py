# Deformation field for GenHead and Portrait4D

import numpy as np
from omegaconf import OmegaConf
import pickle

import torch
import torch.nn.functional as F
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
import dnnlib
from pytorch3d.io import load_obj
from pytorch3d.ops.knn import knn_points,knn_gather

from models.FLAME.DecaFLAME import FLAME
from training.deformer.mesh_renderer import MeshRenderer
from training.deformer.deform_utils import point_to_face_coord, get_face_normal, get_vertex_normal, cam_world_matrix_transform, get_face_tri, get_barycentric

# FLAME-derived 3D deformation field for GenHead
@persistence.persistent_class
class FlameDeformationModule(torch.nn.Module):
    def __init__(self,
        cfg_path,
        flame_full=False,
        box_warp=1.2
    ):
        super().__init__()
        with open(cfg_path, "r") as f:
            conf = OmegaConf.load(f)
        flame_conf = conf.coarse.model
        
        # load basic FLAME model
        self.flame_model = FLAME(flame_conf,flame_full=flame_full)
        
        # rescale FLAME to match the scale of EG3D
        self.flame_model.v_template *= 3 
        self.flame_model.shapedirs *= 3
        self.flame_model.posedirs *= 3
        
        # load canonical face parameters
        ca_shape_param = torch.tensor(np.loadtxt(flame_conf.ca_shape_param).astype(np.float32)).reshape(1,-1)
        ca_exp_param = torch.tensor(np.loadtxt(flame_conf.ca_exp_param).astype(np.float32)).reshape(1,-1)
        ca_pose_param = torch.tensor(np.loadtxt(flame_conf.ca_pose_param).astype(np.float32)).reshape(1,-1)
        
        if flame_full:
            ca_shape_param = torch.cat([ca_shape_param,torch.zeros([1,300-ca_shape_param.shape[1]])],dim=-1)
            ca_exp_param = torch.cat([ca_exp_param,torch.zeros([1,100-ca_exp_param.shape[1]])],dim=-1)
        
        # bounding box scale of the deformation field
        self.box_warp = box_warp
    
        # load face triangles
        with open(flame_conf.flame_mask_pkl, 'rb') as f:
            mask = pickle.load(f, encoding='iso-8859-1')
            face_mask = np.concatenate([
                mask['face'], 
                mask['right_eyeball'], 
                mask['left_eyeball'],
                # mask['eye_region'],
                # mask['forehead'],
            ], 0)
            # face_mask = np.arange(5023)
            face_mask = np.unique(face_mask)
            face_mask = np.sort(face_mask)
            eye_mask = np.concatenate([
                mask['right_eyeball'], 
                mask['left_eyeball'],
            ], 0)
            eye_mask = np.unique(eye_mask)
            eye_mask = np.sort(eye_mask)
            face_wo_eye_mask = mask['face']
            face_wo_eye_mask = np.unique(face_wo_eye_mask)
            face_wo_eye_mask = np.sort(face_wo_eye_mask)
            lips_mask = mask['lips'] 
            noinmouth_mask = mask['noinmouth']
        
        # define canonical face shape
        with torch.no_grad():
            canonical_shape, _, _, canonical_joints = self.flame_model(ca_shape_param,ca_exp_param,ca_pose_param)

            # re-center flame model with canonical neck joint
            canonical_shape = canonical_shape - canonical_joints[:,1:2]
            self.flame_model.v_template = self.flame_model.v_template - canonical_joints[0,1:2]
            canonical_shape_face_only = canonical_shape[:, face_mask]
            
        self.face_mask = face_mask
        self.eye_mask = eye_mask
        self.lips_mask = lips_mask
        self.face_wo_eye_mask = face_wo_eye_mask
        self.noinmouth_mask = noinmouth_mask
        masked_faces = get_face_tri(self.face_mask, self.flame_model.faces_tensor) # get face triangles for masked region
        self.register_buffer('masked_faces', masked_faces)
        masked_eye_faces = get_face_tri(self.eye_mask, self.flame_model.faces_tensor)
        self.register_buffer('masked_eye_faces', masked_eye_faces)
        masked_lips_faces =  get_face_tri(self.lips_mask, self.flame_model.faces_tensor)
        self.register_buffer('masked_lips_faces', masked_lips_faces)
        masked_face_wo_eye_faces = get_face_tri(self.face_wo_eye_mask, self.flame_model.faces_tensor)
        self.register_buffer('masked_face_wo_eye_faces', masked_face_wo_eye_faces)
        masked_noinmouth_faces =  get_face_tri(self.noinmouth_mask, self.flame_model.faces_tensor)
        self.register_buffer('masked_noinmouth_faces', masked_noinmouth_faces)

        self.register_buffer('canonical_shape', canonical_shape)
        self.register_buffer('canonical_shape_face_only', canonical_shape_face_only)
        
        # load info for simplified FLAME mesh
        simplify_barycentric = np.load(flame_conf.simplify_baricentric, allow_pickle=True)
        self.register_buffer('simplify_idxs',torch.from_numpy(simplify_barycentric.item()['idxs']))
        self.register_buffer('simplify_baris',torch.from_numpy(simplify_barycentric.item()['baris']))
        self.register_buffer('simplify_faces',torch.from_numpy(simplify_barycentric.item()['faces']))
        self.register_buffer('simplify_faces_region_idxs',torch.from_numpy(simplify_barycentric.item()['coarse_face_region_idxs']))
        canonical_shape_coarse = torch.sum(canonical_shape[:,self.simplify_idxs]*self.simplify_baris.unsqueeze(0).unsqueeze(-1),dim=-2)
        self.register_buffer('canonical_shape_coarse', canonical_shape_coarse)
        
        # load info for small face region
        add_info = np.load(flame_conf.addition_info,allow_pickle=True)
        self.register_buffer('eye_idxs',torch.from_numpy(add_info.item()['eye_idxs']))
        self.register_buffer('face_idxs',torch.from_numpy(add_info.item()['face_idxs']))
        self.register_buffer('lips_idxs',torch.from_numpy(lips_mask))
        self.register_buffer('noinmouth_idxs',torch.from_numpy(noinmouth_mask))
        self.register_buffer('vertices_region_idxs',torch.from_numpy(add_info.item()['vertices_region_idxs']))
        self.register_buffer('dynamic_idxs',torch.from_numpy(add_info.item()['dynamic_idxs']))
        smallface_idxs = add_info.item()['smallface_idxs']
        masked_smallface_faces = get_face_tri(smallface_idxs, self.flame_model.faces_tensor)
        self.register_buffer('smallface_idxs',torch.from_numpy(smallface_idxs))
        self.register_buffer('masked_smallface_faces', masked_smallface_faces)
        
        # load info for eyelids region
        eyelids_idxs = add_info.item()['eyelids_idxs']
        masked_eyelids_faces = get_face_tri(eyelids_idxs, self.flame_model.faces_tensor)
        self.register_buffer('eyelids_idxs',torch.from_numpy(eyelids_idxs))
        self.register_buffer('masked_eyelids_faces', masked_eyelids_faces)

        eye_mask = torch.from_numpy(add_info.item()['eye_mask'])
        eye_mask_face_simplify = (torch.sum(eye_mask[self.simplify_idxs[self.simplify_faces]].reshape(len(self.simplify_faces),-1),-1)!=0)
        self.register_buffer('if_eye_face_simplify',eye_mask_face_simplify)
        
        # mesh renderer
        self.mesh_renderer = MeshRenderer()
    
    @torch.no_grad()
    def get_vertex_normal(self,vertices,faces):
        return get_vertex_normal(vertices,faces)

    @torch.no_grad()
    def renderer_visualize(self,shape_params,exp_params,pose_params,eye_pose_params,c,fov=12,half_size=128,only_face=False,face_woeye=False,eye_mask=False):
        B = shape_params.shape[0]
        target_shape, _, _, _ = self.flame_model(shape_params,exp_params,pose_params,eye_pose_params)
        if only_face:
            target_shape = target_shape[:, self.face_mask]
            vertex_feature = self.canonical_shape_face_only.repeat(B,1,1)/(3*0.2)  # 4 to match training uv scale
            faces = self.masked_faces.unsqueeze(0).repeat(B,1,1)
        elif face_woeye:
            target_shape = target_shape[:, self.face_wo_eye_mask]
            vertex_feature = torch.ones_like(target_shape)
            faces = self.masked_face_wo_eye_faces.unsqueeze(0).repeat(B,1,1)
        elif eye_mask:
            target_shape = target_shape[:, self.eye_mask]
            vertex_feature = torch.ones_like(target_shape)
            faces = self.masked_eye_faces.unsqueeze(0).repeat(B,1,1)
        else:
            faces = self.flame_model.faces_tensor.unsqueeze(0).repeat(B,1,1)
            vertex_feature = get_vertex_normal(self.canonical_shape.repeat(B,1,1),faces)
            # vertex_feature = torch.zeros_like(target_shape)
            # vertex_feature[:, 0] = 0.5

        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        world2cam_matrix = cam_world_matrix_transform(cam2world_matrix)

        points_homogeneous = torch.ones((target_shape.shape[0], target_shape.shape[1], target_shape.shape[2] + 1), device=target_shape.device)
        points_homogeneous[:, :, :3] = target_shape

        cam_target_shape = torch.bmm(world2cam_matrix, points_homogeneous.permute(0,2,1)).permute(0,2,1) # (B,N,4)
        cam_target_shape = cam_target_shape[...,:3]
        cam_target_shape[..., -1] = -cam_target_shape[..., -1] # reverse z axis to fit pytorch3d renderer

        mask, depth, uv = self.mesh_renderer(fov, half_size*2, cam_target_shape, faces, vertex_feature)

        return uv, mask
    
    # renderer for FLAME mesh
    @torch.no_grad()
    def renderer(self,shape_params,exp_params,pose_params,eye_pose_params,c,fov=12,half_size=32,eye_blink_params=None,only_face=True,small_face=False,cull_backfaces=True,\
        custom_feature=None,eye_mask=False,face_woeye=False,noinmouth=False,use_rotation_limits=False):
        
        B = shape_params.shape[0]
        
        # target flame shape to be rendered
        target_shape, _, target_landmarks3d, _ = self.flame_model(shape_params,exp_params,pose_params,eye_pose_params,use_rotation_limits=use_rotation_limits)
        target_shape_full = target_shape.detach().clone()
        
        # use separate expression parameter for eye blink
        if eye_blink_params is not None:
            target_shape_eye_blink, _, _, _ = self.flame_model(shape_params,eye_blink_params,pose_params,eye_pose_params,use_rotation_limits=use_rotation_limits)
            target_shape[:,self.eyelids_idxs] = target_shape_eye_blink[:,self.eyelids_idxs]
            
        if face_woeye: # render without eyeballs
            target_shape = target_shape[:, self.face_wo_eye_mask]
            vertex_feature = torch.ones_like(target_shape)
            faces = self.masked_face_wo_eye_faces.unsqueeze(0).repeat(B,1,1)
        elif eye_mask: # render only eye region
            target_shape = target_shape[:, self.eye_mask]
            vertex_feature = torch.ones_like(target_shape)
            faces = self.masked_eye_faces.unsqueeze(0).repeat(B,1,1)
        elif only_face: # render uv coordinates for face region
            target_shape = target_shape[:, self.face_mask]
            vertex_feature = self.canonical_shape.repeat(B,1,1)/(3*0.2) # rescale coordinates to increase image contrast
            vertex_feature[:,self.eye_mask,2] = vertex_feature[:,self.eye_mask,2] - 0.5 # change eye region color
            vertex_feature = vertex_feature[:, self.face_mask]
            faces = self.masked_faces.unsqueeze(0).repeat(B,1,1)
        elif small_face: # render only a cropped small face region
            target_shape = target_shape[:, self.smallface_idxs]
            vertex_feature = torch.ones_like(target_shape)
            faces = self.masked_smallface_faces.unsqueeze(0).repeat(B,1,1)
        elif noinmouth: # render without inner mouth
            target_shape = target_shape[:, self.noinmouth_mask]
            vertex_feature = torch.ones_like(target_shape)
            faces = self.masked_noinmouth_faces.unsqueeze(0).repeat(B,1,1)
        elif custom_feature is not None: # render with custom feature color
            vertex_feature = custom_feature
            faces = self.flame_model.faces_tensor.unsqueeze(0).repeat(B,1,1)
        else: # render whole face region using simplified flame mesh
            target_shape = torch.sum(target_shape[:,self.simplify_idxs]*self.simplify_baris.unsqueeze(0).unsqueeze(-1),dim=-2)
            vertex_feature = self.canonical_shape_coarse.repeat(B,1,1)/(3*0.2)
            faces = self.simplify_faces.unsqueeze(0).repeat(B,1,1)
        

        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        world2cam_matrix = cam_world_matrix_transform(cam2world_matrix)

        points_homogeneous = torch.ones((target_shape.shape[0], target_shape.shape[1], target_shape.shape[2] + 1), device=target_shape.device)
        points_homogeneous[:, :, :3] = target_shape
        
        # target shape in camera coordinates
        cam_target_shape = torch.bmm(world2cam_matrix, points_homogeneous.permute(0,2,1)).permute(0,2,1) # (B,N,4)
        cam_target_shape = cam_target_shape[...,:3]
        cam_target_shape[..., -1] = -cam_target_shape[..., -1] # reverse z axis to align with pytorch3d renderer

        mask, depth, uv = self.mesh_renderer(fov, half_size*2, cam_target_shape, faces, vertex_feature,cull_backfaces=cull_backfaces)

        #-------------------------------------------
        # for 2D landmarks rendering

        points_homogeneous_landmarks = torch.ones((target_landmarks3d.shape[0], target_landmarks3d.shape[1], target_landmarks3d.shape[2] + 1), device=target_landmarks3d.device)
        points_homogeneous_landmarks[:, :, :3] = target_landmarks3d.clone()

        cam_target_shape_landmarks = torch.bmm(world2cam_matrix, points_homogeneous_landmarks.permute(0,2,1)).permute(0,2,1) # (B,N,4)
        cam_target_shape_landmarks = cam_target_shape_landmarks[...,:3]
        cam_target_shape_landmarks[..., -1] = -cam_target_shape_landmarks[..., -1] # reverse z axis to fit pytorch3d renderer

        K = c[:,16:25].detach().clone().view(-1, 3, 3)
        K[:,:2] *= half_size*2
        project_target_shape_landmarks = torch.bmm(K, cam_target_shape_landmarks.permute(0, 2, 1)).permute(0, 2, 1)
        project_target_shape_landmarks /= project_target_shape_landmarks[..., -1:].clone()
        landmarks_2d = project_target_shape_landmarks[...,:2]

        return uv, mask, depth, landmarks_2d
    
    # decompose the head pose represented fully by camera pose to neck pose and a new camera pose, i.e., c_compose = neck pose + c_decompose.
    # the obtained neck pose and c_decompose ensure that the neck joint position in the camera space stays unchanged w.r.t.
    # its original position with zero neck pose and c_compose
    @torch.no_grad()
    def decompose_camera_pose(self,c_compose,shape_params,exp_params,pose_params,use_rotation_limits=False):
        
        # obtain 3D neck joint position using zero neck pose 
        _, _, _, target_joints = self.flame_model(shape_params,exp_params,torch.cat([torch.zeros_like(pose_params[:,:3]),pose_params[:,3:]],dim=1),torch.zeros_like(pose_params),use_rotation_limits=use_rotation_limits)
        target_joints = target_joints[:,1:2,:]

        cam2world_matrix = c_compose[:, :16].view(-1, 4, 4)
        world2cam_matrix = cam_world_matrix_transform(cam2world_matrix)
        points_homogeneous = torch.ones((target_joints.shape[0], target_joints.shape[1], target_joints.shape[2] + 1), device=target_joints.device)
        points_homogeneous[:, :, :3] = target_joints
        
        # original neck joint position in the camera space
        cam_target_joint = torch.bmm(world2cam_matrix, points_homogeneous.permute(0,2,1)).permute(0,2,1) # (B,N,4)
        
        # obtain 3D neck joint position using given neck pose 
        _, _, _, target_joints_decompose = self.flame_model(shape_params,exp_params,pose_params,torch.zeros_like(pose_params),use_rotation_limits=use_rotation_limits)
        
        # calculate new camera matrix given the difference between the two neck joint positions
        headpose_rot = self.flame_model._pose2rot(pose_params[:,:3]) 
        world2cam_rot_decompose = world2cam_matrix[:,:3,:3] @ headpose_rot.permute(0,2,1) # remove rotation derived from neck pose
        world2cam_trans_decompose = cam_target_joint[:,:,:3].permute(0,2,1) - torch.bmm(world2cam_rot_decompose, target_joints_decompose[:,1:2,:].permute(0,2,1)) # calculate camera space translation

        cam2world_matrix_decompose = torch.eye(4,device=c_compose.device).unsqueeze(0).repeat(c_compose.shape[0], 1, 1)
        cam2world_rot_decompose = world2cam_rot_decompose.permute(0,2,1)
        cam2world_trans_decompose = - cam2world_rot_decompose @ world2cam_trans_decompose
        cam2world_matrix_decompose[:,:3,:3] = cam2world_rot_decompose
        cam2world_matrix_decompose[:,:3,3:] = cam2world_trans_decompose

        c_decompose = torch.cat([cam2world_matrix_decompose.view(-1, 16),c_compose[:, 16:25]],dim=-1)

        return c_decompose
    
    # derive 3D deformation field from FLAME mesh using closest point method:
    # for a point in the 3D space, its deformation is copied from that of its closest point on the mesh surface
    @torch.no_grad()
    def closest_point_deformation(self,x_t,shape_params,exp_params,pose_params,eye_pose_params, simplify=False, use_rotation_limits=False):
        B = len(x_t)
        
        # target FLAME mesh for deformation field derivation
        target_shape, _, _, target_joints = self.flame_model(shape_params,exp_params,pose_params,eye_pose_params, use_rotation_limits=use_rotation_limits) # target_shape (B,N,3)
        faces = self.flame_model.faces_tensor.unsqueeze(0).repeat(B,1,1) # faces (B,T,3)
        canonical_shape = self.canonical_shape.repeat(B,1,1)
        if simplify:
            target_shape = torch.sum(target_shape[:,self.simplify_idxs]*self.simplify_baris.unsqueeze(0).unsqueeze(-1),dim=-2)
            faces = self.simplify_faces.unsqueeze(0).repeat(B,1,1)
            canonical_shape = self.canonical_shape_coarse.repeat(B,1,1)
            
        # point-to-point distance and corresponding nearest vertice indices for x_t 
        dists, nearest_idx,_ = knn_points(x_t,target_shape,K=3) # return squared distance
        inv_dists = 1/(torch.sqrt(dists+1e-12))
        weights = inv_dists/torch.sum(inv_dists,dim=-1,keepdim=True)
        weights = weights.unsqueeze(-1)
        
        # calculate deformation using weighted average of that of the nearest mesh vertices
        target_closest_pts = knn_gather(target_shape,nearest_idx)
        canonical_closest_pts = knn_gather(canonical_shape,nearest_idx)
        offset = torch.sum((canonical_closest_pts-target_closest_pts)*weights,dim=-2)
        
        # deformed points in the canonical space
        x_c = x_t + offset

        # return x_c, torch.sqrt(dists[...,0]+1e-12)
        return {'x_c':x_c, 'dist':torch.sqrt(dists[...,0]+1e-12), 'weights': None, 'closest_vts_region': None}
    
    
    # derive 3D deformation field from FLAME mesh using closest point method (separate deformation for face, eyes, and mouth):
    @torch.no_grad()
    def closest_point_deformation_separate(self,x_t,shape_params,exp_params,pose_params,eye_pose_params,eye_blink_params=None,use_rotation_limits=False,mouth=False):
        B = len(x_t)
        target_shape, _, _, target_joints = self.flame_model(shape_params,exp_params,pose_params,eye_pose_params,use_rotation_limits=use_rotation_limits) # target_shape (B,N,3)
        if eye_blink_params is not None:
            target_shape_eye_blink, _, _, _ = self.flame_model(shape_params,eye_blink_params,pose_params,eye_pose_params,use_rotation_limits=use_rotation_limits)
            target_shape[:,self.eyelids_idxs] = target_shape_eye_blink[:,self.eyelids_idxs]
            
        # set expression to zero for mouth deformation to model teeth movement
        if mouth:
            target_shape_noexp_noeye, _, _, target_joints = self.flame_model(shape_params,torch.zeros_like(exp_params),pose_params,torch.zeros_like(eye_pose_params),use_rotation_limits=use_rotation_limits) # target_shape (B,N,3)
        
        faces = self.flame_model.faces_tensor.unsqueeze(0).repeat(B,1,1) # faces (B,T,3)
        canonical_shape = self.canonical_shape.repeat(B,1,1)
        target_shape_eye = target_shape[:,self.eye_idxs]
        target_shape_face = target_shape[:,self.face_idxs]
        if mouth:
            target_shape_mouth = target_shape_noexp_noeye[:,self.lips_idxs]

        target_shape_eye_left = target_shape_eye[:, :len(self.eye_idxs)//2]
        target_shape_eye_right = target_shape_eye[:, len(self.eye_idxs)//2:]
        center_eye_left = target_shape_eye_left.mean(dim=1, keepdim=True)
        size_eye_left = (target_shape_eye_left.max(dim=1, keepdim=True)[0] - target_shape_eye_left.min(dim=1, keepdim=True)[0]) * 1.2
        inside_bbox_eye_left = ((x_t - center_eye_left).abs() < (size_eye_left * 0.5)).all(dim=-1)
        center_eye_right = target_shape_eye_right.mean(dim=1, keepdim=True)
        size_eye_right = (target_shape_eye_right.max(dim=1, keepdim=True)[0] - target_shape_eye_right.min(dim=1, keepdim=True)[0]) * 1.2
        inside_bbox_eye_right = ((x_t - center_eye_right).abs() < (size_eye_right * 0.5)).all(dim=-1)
        inside_bbox_eye =  (inside_bbox_eye_left + inside_bbox_eye_right).float()

        closest_dist_eye, nearest_idx_eye,_ = knn_points(x_t,target_shape_eye,K=1) # (B,M,1)
        closest_dist_face, nearest_idx_face,_ = knn_points(x_t,target_shape_face,K=1) # (B,M,1)
        
        if mouth:
            closest_dist_mouth, nearest_idx_mouth,_ = knn_points(x_t,target_shape_mouth,K=1) # (B,M,1)
            lips_idxs_batch = self.lips_idxs[None, None].repeat(nearest_idx_mouth.shape[0], nearest_idx_mouth.shape[1], 1)
            nearest_idx_mouth = torch.gather(lips_idxs_batch, 2, nearest_idx_mouth)
        else:
            closest_dist_mouth = torch.zeros_like(closest_dist_eye)

        nearest_region_idx_eye = knn_gather(self.vertices_region_idxs.unsqueeze(0).repeat(B,1,1),nearest_idx_eye+self.eye_idxs[0]).squeeze(-2) # (B,M,max_region_number)
        nearest_region_idx_face = knn_gather(self.vertices_region_idxs.unsqueeze(0).repeat(B,1,1),nearest_idx_face).squeeze(-2) # (B,M,max_region_number)
        if mouth:
            nearest_region_idx_mouth = knn_gather(self.vertices_region_idxs.unsqueeze(0).repeat(B,1,1),nearest_idx_mouth).squeeze(-2) # (B,M,max_region_number)

        target_shape_aug = torch.cat([target_shape,1e3*torch.ones_like(target_shape[:,:1])],dim=1) # augment the target shape with an infinite point
        canonical_shape_aug = torch.cat([canonical_shape,1e3*torch.ones_like(target_shape[:,:1])],dim=1)
        if mouth:
            target_shape_noexp_noeye_aug = torch.cat([target_shape_noexp_noeye,1e3*torch.ones_like(target_shape_noexp_noeye[:,:1])],dim=1)

        target_closest_pts_eye = knn_gather(target_shape_aug,nearest_region_idx_eye) # (B,M,max_region_number,3)
        target_closest_pts_face = knn_gather(target_shape_aug,nearest_region_idx_face) # (B,M,max_region_number,3)
        if mouth:
            target_closest_pts_mouth = knn_gather(target_shape_noexp_noeye_aug,nearest_region_idx_mouth) # (B,M,max_region_number,3)
        canonical_closest_pts_eye = knn_gather(canonical_shape_aug,nearest_region_idx_eye)
        canonical_closest_pts_face = knn_gather(canonical_shape_aug,nearest_region_idx_face)
        if mouth:
            canonical_closest_pts_mouth = knn_gather(canonical_shape_aug,nearest_region_idx_mouth)

        dists_eye = torch.sum((x_t.unsqueeze(-2) - target_closest_pts_eye)**2,-1)
        inv_dists_eye = 1/(dists_eye+1e-10)
        dists_face = torch.sum((x_t.unsqueeze(-2) - target_closest_pts_face)**2,-1)
        inv_dists_face = 1/(dists_face+1e-10)
        if mouth:
            dists_mouth = torch.sum((x_t.unsqueeze(-2) - target_closest_pts_mouth)**2,-1)
            inv_dists_mouth = 1/(dists_mouth+1e-10)

        weights_eye = inv_dists_eye/torch.sum(inv_dists_eye,dim=-1,keepdim=True)
        weights_eye = weights_eye.unsqueeze(-1)

        weights_face = inv_dists_face/torch.sum(inv_dists_face,dim=-1,keepdim=True)
        weights_face = weights_face.unsqueeze(-1)
        if mouth:
            weights_mouth = inv_dists_mouth/torch.sum(inv_dists_mouth,dim=-1,keepdim=True)
            weights_mouth = weights_mouth.unsqueeze(-1)

        offset_eye = torch.sum((canonical_closest_pts_eye-target_closest_pts_eye)*weights_eye,dim=-2) # (B,M,3)
        offset_face = torch.sum((canonical_closest_pts_face-target_closest_pts_face)*weights_face,dim=-2) # (B,M,3)
        if mouth:
            offset_mouth = torch.sum((canonical_closest_pts_mouth-target_closest_pts_mouth)*weights_mouth,dim=-2) # (B,M,3)
            
        # derive canonical position for face, eyes, and mouth, respectively
        x_c_eye = x_t + offset_eye
        x_c_face = x_t + offset_face
        if mouth:
            x_c_mouth = x_t + offset_mouth
        else:
            x_c_mouth = torch.zeros_like(x_t)

        return {'x_c_eye':x_c_eye, 'x_c_face':x_c_face, 'x_c_mouth':x_c_mouth, 'dist_eye':torch.sqrt(closest_dist_eye[...,0]+1e-12), \
            'dist_face':torch.sqrt(closest_dist_face[...,0]+1e-12), 'dist_mouth':torch.sqrt(closest_dist_mouth[...,0]+1e-12), 'inside_bbox_eye':inside_bbox_eye, \
            'closest_vts_region':nearest_region_idx_face,'weights':weights_face}
    
    # derive 3D deformation field from FLAME mesh using closest surface method:
    # for a point in the 3D space, its deformation is calculated from that of its neighboring triangles on the mesh
    @torch.no_grad()
    def closest_surface_deformation(self,x_t,shape_params,exp_params,pose_params,eye_pose_params, use_rotation_limits=False):

        B = len(x_t)

        # get target FLAME shape
        target_shape, _, _, target_joints = self.flame_model(shape_params,exp_params,pose_params,eye_pose_params,use_rotation_limits=use_rotation_limits) # target_shape (B,N,3)
        faces = self.flame_model.faces_tensor.unsqueeze(0).repeat(B,1,1) # faces (B,T,3)
        target_shape_simplify = torch.sum(target_shape[:,self.simplify_idxs]*self.simplify_baris.unsqueeze(0).unsqueeze(-1),dim=-2)
        faces_simplify = self.simplify_faces.unsqueeze(0).repeat(B,1,1)

        canonical_shape = self.canonical_shape.repeat(B,1,1)
        
        # calculate nearest triangles using simplified FLAME mesh
        dists, idxs, baries = point_to_face_coord(x_t,target_shape_simplify,faces_simplify)
        
        # get neighboring vertices based on the connected domain of the nearest triangle
        closest_vts_region = self.simplify_faces_region_idxs[idxs.squeeze(-1)]  #(B,M,50)

        target_shape_aug = torch.cat([target_shape,1e3*torch.ones_like(target_shape[:,:1])],dim=1) # augment the target shape with an infinite point
        canonical_shape_aug = torch.cat([canonical_shape,1e3*torch.ones_like(target_shape[:,:1])],dim=1)

        target_closest_pts = knn_gather(target_shape_aug,closest_vts_region) # (B,M,50,3)
        canonical_closest_pts = knn_gather(canonical_shape_aug,closest_vts_region)

        dists_pts = torch.sum((x_t.unsqueeze(-2) - target_closest_pts)**2,-1)
        inv_dists = 1/(dists_pts+1e-12)
        weights = inv_dists/torch.sum(inv_dists,dim=-1,keepdim=True)
        weights = weights.unsqueeze(-1)
    
        offset = torch.sum((canonical_closest_pts-target_closest_pts)*weights,dim=-2)
        x_c = x_t + offset

        return {'x_c':x_c, 'dist':torch.sqrt(dists+1e-12), 'closest_vts_region':closest_vts_region, 'weights':weights}
    
    # derive 3D deformation field from FLAME mesh using Surface Field defined in GNARF: http://www.computationalimaging.org/publications/gnarf/
    @torch.no_grad()
    def surface_field_deformation(self,x_t,shape_params,exp_params,pose_params,eye_pose_params, simplify=False, only_headpose=False, return_offset=False, use_rotation_limits=False):
        # x_t (B,P,3)
        B = len(x_t)

        # get target flame shape
        target_shape, _, _, target_joints = self.flame_model(shape_params,exp_params,pose_params,eye_pose_params,use_rotation_limits=use_rotation_limits) # target_shape (B,N,3)
        faces = self.flame_model.faces_tensor.unsqueeze(0).repeat(B,1,1) # faces (B,T,3)
        if simplify:
            target_shape = torch.sum(target_shape[:,self.simplify_idxs]*self.simplify_baris.unsqueeze(0).unsqueeze(-1),dim=-2)
            faces = self.simplify_faces.unsqueeze(0).repeat(B,1,1)

        # calculate closest triangles and barycentric coordinates for target space 3d points
        dists, idxs, baries = point_to_face_coord(x_t,target_shape,faces)
        

        # get vertex positions for target and canonical space triangles
        target_tris = torch.stack([target_shape[i][faces[i]] for i in range(B)],0) # (B,T,3,3)
        
        if only_headpose:
            canonical_shape, _, _, _ = self.flame_model(shape_params,exp_params,torch.cat([torch.zeros_like(pose_params[...,:3]),pose_params[...,3:]],dim=-1),eye_pose_params,use_rotation_limits=use_rotation_limits)
        else:
            canonical_shape = self.canonical_shape

        if not simplify:
            canonical_tris = canonical_shape[:,self.flame_model.faces_tensor].repeat(B,1,1,1)
        else:
            canonical_shape_coarse = torch.sum(canonical_shape[:,self.simplify_idxs]*self.simplify_baris.unsqueeze(0).unsqueeze(-1),dim=-2)
            canonical_tris = canonical_shape_coarse[:,self.simplify_faces].repeat(B,1,1,1)

        # calculate face normals
        target_normals = get_face_normal(target_tris)
        canonical_normals = get_face_normal(canonical_tris)

        # surface field calculation from http://www.computationalimaging.org/publications/gnarf/
        # x_c = tri_c*bary + <x_t - tri_tar*bary,n_tar>*n_c

        # vertex positions and normals for the closest triangles in the canonical space
        close_canonical_tris = torch.stack([canonical_tris[i][idxs[i]] for i in range(B)],0)
        close_canonical_normals = torch.stack([canonical_normals[i][idxs[i]] for i in range(B)],0)
        # canonical space closest surface points
        surface_c =  torch.sum(close_canonical_tris*baries.unsqueeze(-1),dim=-2)

        # vertex positions and normals for the closest triangles in the target space
        close_target_tris = torch.stack([target_tris[i][idxs[i]] for i in range(B)],0)
        close_target_normals = torch.stack([target_normals[i][idxs[i]] for i in range(B)],0)
        # 3d offsets from canonical surface points to the deformed 3d points in the canonical space
        offset_c = x_t - torch.sum(close_target_tris*baries.unsqueeze(-1),dim=-2)
        offset_c = torch.sum(offset_c*close_target_normals,dim=-1,keepdim=True)*close_canonical_normals

        # final 3d position in the canonical space for the input target points
        # offset = surface_c + offset_c - x_t
        x_c = surface_c + offset_c
        if return_offset:
            return x_c - x_t, dists

        return x_c, dists

    # get surface field deformation for pre-defined voxel grids
    @torch.no_grad()
    def get_grid_deformation(self,shape_params,exp_params,pose_params,eye_pose_params,resolution=16,simplify=False,only_headpose=False,return_offset=False, use_rotation_limits=False):
        grids = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=shape_params.device), torch.arange(resolution, dtype=torch.float32, device=shape_params.device), torch.arange(resolution, dtype=torch.float32, device=shape_params.device), indexing='ij'),dim=-1) * (1./resolution) + (0.5/resolution) - 0.5 #[-0.5,0.5]
        grids *= self.box_warp
        grids = grids.reshape(1,resolution**3,3).repeat(shape_params.shape[0],1,1)

        grids_c, dists = self.surface_field_deformation(grids, shape_params,exp_params,pose_params,eye_pose_params,simplify=simplify,only_headpose=only_headpose,return_offset=return_offset, use_rotation_limits=use_rotation_limits)

        return grids.permute(0,2,1).reshape(-1,3,resolution,resolution,resolution), grids_c.permute(0,2,1).reshape(-1,3,resolution,resolution,resolution), dists.reshape(-1,1,resolution,resolution,resolution)
    
    # from world-space coordinates to local coordinates of the grids
    @torch.no_grad()
    def world2grid(self,inputs,grids):
        grid_x = (inputs[...,0] - grids[:,0,None,-1,-1,-1])/(grids[:,0,None,-1,-1,-1]-grids[:,0,None,0,0,0])*2 + 1
        grid_y = (inputs[...,1] - grids[:,1,None,-1,-1,-1])/(grids[:,1,None,-1,-1,-1]-grids[:,1,None,0,0,0])*2 + 1
        grid_z = (inputs[...,2] - grids[:,2,None,-1,-1,-1])/(grids[:,2,None,-1,-1,-1]-grids[:,2,None,0,0,0])*2 + 1

        return torch.stack([grid_x,grid_y,grid_z],dim=-1)
    
    # render deformation flow onto three orthogonal planes (xy, yz, zx)
    @torch.no_grad()
    def render_orthogonal_flow(self,shape_params,exp_params,pose_params,eye_pose_params, half_size=128, only_face=False,cull_backfaces=True,use_rotation_limits=False, scale=1.0):
        B = shape_params.shape[0]
        
        # shape with no head pose
        target_shape, _, _, _ = self.flame_model(shape_params,exp_params,torch.cat([torch.zeros_like(pose_params[:,:3]),pose_params[:,3:]],dim=-1),eye_pose_params,use_rotation_limits=use_rotation_limits)
        ca_pose = torch.tensor([[0,0,0,0.3,0,0]],dtype=pose_params.dtype).to(pose_params.device).repeat(B,1)
        canonical_shape, _, _, _ = self.flame_model(shape_params,torch.zeros_like(exp_params),ca_pose,torch.zeros_like(eye_pose_params),use_rotation_limits=use_rotation_limits)
        
        flow_xyz = canonical_shape - target_shape
        flow_xyz *= scale # match the scale of tri-plane features

        target_shape = torch.cat([target_shape,target_shape[...,[1,2,0]],target_shape[...,[2,0,1]]],dim=0)
        faces = self.flame_model.faces_tensor.unsqueeze(0).repeat(3*B,1,1)

        flow_xyz = torch.cat([flow_xyz[...,[0,1]],flow_xyz[...,[1,2]],flow_xyz[...,[2,0]]],dim=0)

        mask, depth, flow = self.mesh_renderer(None, half_size*2, target_shape, faces, flow_xyz,cull_backfaces=cull_backfaces,perspective=False,scale=scale)

        flow = flow.flip(-2)
        mask = mask.flip(-2)

        return flow.reshape(B,3,-1,flow.shape[-2],flow.shape[-1]), mask.reshape(B,3,-1,mask.shape[-2],mask.shape[-1])
    
    # render shape deformation flow onto three orthogonal planes (xy, yz, zx)
    @torch.no_grad()
    def render_orthogonal_shape_flow(self,shape_params,exp_params,pose_params,eye_pose_params, half_size=128, only_face=False,cull_backfaces=True,use_rotation_limits=False, scale=1.0):
        B = shape_params.shape[0]
        
        # shape without expression, head pose, and eye gaze
        ca_pose = torch.tensor([[0,0,0,0.3,0,0]],dtype=pose_params.dtype).to(pose_params.device).repeat(B,1)
        target_shape, _, _, _ = self.flame_model(shape_params,torch.zeros_like(exp_params), ca_pose, torch.zeros_like(eye_pose_params),use_rotation_limits=use_rotation_limits)
        canonical_shape, _, _, _ = self.flame_model(torch.zeros_like(shape_params),torch.zeros_like(exp_params),ca_pose,torch.zeros_like(eye_pose_params),use_rotation_limits=use_rotation_limits)
        
        flow_xyz = canonical_shape - target_shape
        flow_xyz *= scale # match the scale of tri-plane features

        target_shape = torch.cat([target_shape,target_shape[...,[1,2,0]],target_shape[...,[2,0,1]]],dim=0)
        faces = self.flame_model.faces_tensor.unsqueeze(0).repeat(3*B,1,1)

        flow_xyz = torch.cat([flow_xyz[...,[0,1]],flow_xyz[...,[1,2]],flow_xyz[...,[2,0]]],dim=0)

        mask, depth, flow = self.mesh_renderer(None, half_size*2, target_shape, faces, flow_xyz,cull_backfaces=cull_backfaces,perspective=False,scale=scale)

        flow = flow.flip(-2)
        mask = mask.flip(-2)

        return flow.reshape(B,3,-1,flow.shape[-2],flow.shape[-1]), mask.reshape(B,3,-1,mask.shape[-2],mask.shape[-1])
    
    # deform target space 3D point x_t to canonical space for feature acquisition
    def forward(self,x_t,shape_params,exp_params,pose_params,eye_pose_params,eye_blink_params=None, grid_approx=True, simplify=True,only_headpose=True,return_offset=True, use_rotation_limits=False,mouth=False,part=True, smooth_headpose=True, eye_mask=None, mouth_mask=None):
        
        # use pre-defined voxel grids for efficient head pose deformation calculation
        if grid_approx:
            
            # elimiate head pose rotation first
            if not smooth_headpose:
                grids, grids_c, _ = self.get_grid_deformation(shape_params,exp_params,pose_params,eye_pose_params, simplify=simplify,only_headpose=only_headpose,return_offset=return_offset, use_rotation_limits=use_rotation_limits)
                x_grids = self.world2grid(x_t,grids)
                x_grids = torch.flip(x_grids.reshape(x_grids.shape[0],x_grids.shape[1],1,1,3),dims=[-1])

                if not return_offset:
                    x_c = F.grid_sample(grids_c,x_grids,mode='bilinear',padding_mode="border",align_corners=True)
                    x_c = x_c.reshape(-1,3,x_t.shape[1]).permute(0,2,1)
                else:
                    x_c = F.grid_sample(grids_c,x_grids,mode='bilinear',padding_mode="zeros",align_corners=True)
                    x_c = x_c.reshape(-1,3,x_t.shape[1]).permute(0,2,1)
                    x_c = x_c + x_t
            else:
                grids, grids_c, grids_dists = self.get_grid_deformation(shape_params,exp_params,pose_params,eye_pose_params, resolution=24, simplify=simplify,only_headpose=only_headpose,return_offset=return_offset, use_rotation_limits=use_rotation_limits)
                x_grids = self.world2grid(x_t,grids)
                x_grids = torch.flip(x_grids.reshape(x_grids.shape[0],x_grids.shape[1],1,1,3),dims=[-1])
                
                # obtain coarse deformation with low-res grids
                grids_, grids_c_, _ = self.get_grid_deformation(shape_params,exp_params,pose_params,eye_pose_params, resolution=8, simplify=simplify,only_headpose=only_headpose,return_offset=return_offset, use_rotation_limits=use_rotation_limits)
                x_grids_ = self.world2grid(x_t,grids_)
                x_grids_ = torch.flip(x_grids_.reshape(x_grids_.shape[0],x_grids_.shape[1],1,1,3),dims=[-1])

                # smooth for the grid deformation (for discontinuity of headpose deformation)
                filter3d = torch.ones([3,1,3,3,3]).to(x_grids_.device)/27
                grids_c_smooth = F.pad(grids_c_,(1,1,1,1,1,1),'replicate')
                grids_c_smooth = F.conv3d(grids_c_smooth,filter3d,groups=3)

                grids_c_smooth = F.pad(grids_c_smooth,(1,1,1,1,1,1),'replicate')
                grids_c_smooth = F.conv3d(grids_c_smooth,filter3d,groups=3)

                if not return_offset:
                    x_c = F.grid_sample(grids_c,x_grids,mode='bilinear',padding_mode="border",align_corners=True)
                    x_c = x_c.reshape(-1,3,x_t.shape[1]).permute(0,2,1)
                else:
                    x_c = F.grid_sample(grids_c,x_grids,mode='bilinear',padding_mode="zeros",align_corners=True)
                    x_c = x_c.reshape(-1,3,x_t.shape[1]).permute(0,2,1)

                    x_c_smooth = F.grid_sample(grids_c_smooth,x_grids_,mode='bilinear',padding_mode="zeros",align_corners=True)
                    x_c_smooth = x_c_smooth.reshape(-1,3,x_t.shape[1]).permute(0,2,1)

                    x_dists = F.grid_sample(grids_dists,x_grids,mode='bilinear',padding_mode="zeros",align_corners=True)
                    x_dists = x_dists.reshape(-1,1,x_t.shape[1]).permute(0,2,1)

                    weights = torch.exp(-1000*F.relu(x_dists-3e-3))
                    x_c = weights*x_c + (1-weights)*x_c_smooth

                    x_c = x_c + x_t
            
            # canonical position without head pose
            x_c = x_c.detach()
            
            # derive other deformation (shape, expression)
            if only_headpose:
                if not part:
                    deform_out = self.closest_surface_deformation(x_c,shape_params,exp_params,torch.cat([torch.zeros_like(pose_params[...,:3]),pose_params[...,3:]],dim=-1),eye_pose_params, use_rotation_limits=use_rotation_limits)
                else:
                    deform_out = self.closest_point_deformation_separate(x_c,shape_params,exp_params,torch.cat([torch.zeros_like(pose_params[...,:3]),pose_params[...,3:]],dim=-1),eye_pose_params,eye_blink_params=eye_blink_params,use_rotation_limits=use_rotation_limits,mouth=mouth)
        
        # naive deformation calculation using closest point method
        else:
            deform_out = self.closest_point_deformation(x_t,shape_params,exp_params,pose_params,eye_pose_params, simplify=False, use_rotation_limits=use_rotation_limits)

        return {key:deform_out[key].detach() for key in deform_out}

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# FLAME-derived 3D deformation field for Portrait4D (only consider head pose rotation)
@persistence.persistent_class
class FlameDeformationModuleOnlyHead(FlameDeformationModule):
    def __init__(self,
        cfg_path,
        flame_full=False,
        box_warp=1.2
    ):
        super().__init__(cfg_path,flame_full,box_warp)
    
    @torch.no_grad()
    def render_orthogonal_flow(self,shape_params,exp_params,pose_params,eye_pose_params, half_size=128, only_face=False,cull_backfaces=True,use_rotation_limits=False, scale=1.0):
        B = shape_params.shape[0]
        # shape with no head pose
        target_shape, _, _, _ = self.flame_model(shape_params,exp_params,torch.cat([torch.zeros_like(pose_params[:,:3]),pose_params[:,3:]],dim=-1),eye_pose_params,use_rotation_limits=use_rotation_limits)
        ca_pose = torch.tensor([[0,0,0,0.3,0,0]],dtype=pose_params.dtype).to(pose_params.device).repeat(B,1)
        canonical_shape, _, _, _ = self.flame_model(shape_params,torch.zeros_like(exp_params),ca_pose,torch.zeros_like(eye_pose_params),use_rotation_limits=use_rotation_limits)
        
        flow_xyz = canonical_shape - target_shape
        flow_xyz *= scale # match the scale of tri-plane features

        target_shape = torch.stack([target_shape,target_shape[...,[1,2,0]],target_shape[...,[2,0,1]]],dim=1)
        target_shape = target_shape.reshape(B*3,-1,3)
        faces = self.flame_model.faces_tensor.unsqueeze(0).repeat(3*B,1,1)

        flow_xyz = torch.stack([flow_xyz[...,[0,1]],flow_xyz[...,[1,2]],flow_xyz[...,[2,0]]],dim=1)
        flow_xyz = flow_xyz.reshape(B*3,-1,2)

        mask, depth, flow = self.mesh_renderer(None, half_size*2, target_shape, faces, flow_xyz,cull_backfaces=cull_backfaces,perspective=False,scale=scale)

        flow = flow.flip(-2)
        mask = mask.flip(-2)

        return flow.reshape(B,3,-1,flow.shape[-2],flow.shape[-1]), mask.reshape(B,3,-1,mask.shape[-2],mask.shape[-1])
    
    # deform target space 3D point x_t to canonical space for feature acquisition
    def forward(self,x_t,shape_params,exp_params,pose_params,eye_pose_params, simplify=True, use_rotation_limits=False, smooth_headpose=True, smooth_th=3e-3):
        if not smooth_headpose:
            grids, grids_c, _ = self.get_grid_deformation(shape_params,exp_params,pose_params,eye_pose_params, simplify=simplify,only_headpose=True,return_offset=True, use_rotation_limits=use_rotation_limits)
            x_grids = self.world2grid(x_t,grids)
            x_grids = torch.flip(x_grids.reshape(x_grids.shape[0],x_grids.shape[1],1,1,3),dims=[-1])

            x_c = F.grid_sample(grids_c,x_grids,mode='bilinear',padding_mode="zeros",align_corners=True)
            x_c = x_c.reshape(-1,3,x_t.shape[1]).permute(0,2,1)
            x_c = x_c + x_t
        else:
            grids, grids_c, grids_dists = self.get_grid_deformation(shape_params,exp_params,pose_params,eye_pose_params, resolution=24, simplify=simplify,only_headpose=True,return_offset=True, use_rotation_limits=use_rotation_limits)
            x_grids = self.world2grid(x_t,grids)
            x_grids = torch.flip(x_grids.reshape(x_grids.shape[0],x_grids.shape[1],1,1,3),dims=[-1])


            grids_, grids_c_, _ = self.get_grid_deformation(shape_params,exp_params,pose_params,eye_pose_params, resolution=8, simplify=simplify,only_headpose=True,return_offset=True, use_rotation_limits=use_rotation_limits)
            x_grids_ = self.world2grid(x_t,grids_)
            x_grids_ = torch.flip(x_grids_.reshape(x_grids_.shape[0],x_grids_.shape[1],1,1,3),dims=[-1])

            # smooth for the grid deformation (for discontinuity of headpose deformation)
            filter3d = torch.ones([3,1,3,3,3]).to(x_grids_.device)/27
            grids_c_smooth = F.pad(grids_c_,(1,1,1,1,1,1),'replicate')
            grids_c_smooth = F.conv3d(grids_c_smooth,filter3d,groups=3)

            grids_c_smooth = F.pad(grids_c_smooth,(1,1,1,1,1,1),'replicate')
            grids_c_smooth = F.conv3d(grids_c_smooth,filter3d,groups=3)

            x_c = F.grid_sample(grids_c,x_grids,mode='bilinear',padding_mode="zeros",align_corners=True)
            x_c = x_c.reshape(-1,3,x_t.shape[1]).permute(0,2,1)

            x_c_smooth = F.grid_sample(grids_c_smooth,x_grids_,mode='bilinear',padding_mode="zeros",align_corners=True)
            x_c_smooth = x_c_smooth.reshape(-1,3,x_t.shape[1]).permute(0,2,1)

            x_dists = F.grid_sample(grids_dists,x_grids,mode='bilinear',padding_mode="zeros",align_corners=True)
            x_dists = x_dists.reshape(-1,1,x_t.shape[1]).permute(0,2,1)

            weights = torch.exp(-1000*F.relu(x_dists-smooth_th))
            x_c = weights*x_c + (1-weights)*x_c_smooth

            x_c = x_c + x_t
            
        x_c = x_c.detach()

        return {'x_c':x_c}

# Deformation module for GenHead
@persistence.persistent_class
class DeformationModule(torch.nn.Module):
    def __init__(self,
        flame_cfg_path = 'models/FLAME/cfg.yaml',
        flame_full = False,
        dynamic_texture = False, # deprecated
        part = False,
        box_warp = 1.2,
        **kwargs, 
    ):
        super().__init__()
        self.part_model = part
        self.flame_deform = FlameDeformationModule(flame_cfg_path,flame_full=flame_full,box_warp=box_warp)
        self.dynamic_mask_basis = None
    
    def renderer(self, shape_params,exp_params,pose_params,eye_pose_params,camera, eye_blink_params=None, only_face=True,small_face=False,cull_backfaces=True,half_size=32,eye_mask=False,face_woeye=False,noinmouth=False,custom_feature=None,use_rotation_limits=False):
        # return Bx3xHxW correspondence map
        return self.flame_deform.renderer(shape_params,exp_params,pose_params,eye_pose_params,camera, eye_blink_params=eye_blink_params,only_face=only_face,small_face=small_face,cull_backfaces=cull_backfaces,half_size=half_size,eye_mask=eye_mask,face_woeye=face_woeye,noinmouth=noinmouth,custom_feature=custom_feature,use_rotation_limits=use_rotation_limits)
    
    def forward(self, x_t, shape_params,exp_params,pose_params,eye_pose_params, eye_blink_params=None, exp_params_dynamics=None, cache_backbone=False, use_cached_backbone=False, use_rotation_limits=False, mouth=False, eye_mask=None, mouth_mask=None):
        
        # use part-wise deformation field
        if self.part_model:
            with torch.no_grad():
                out = self.flame_deform(x_t, shape_params,exp_params,pose_params,eye_pose_params, eye_blink_params=eye_blink_params, mouth=mouth, use_rotation_limits=use_rotation_limits, part=True, eye_mask=eye_mask, mouth_mask=mouth_mask)
                x_eye_prime = out['x_c_eye']
                x_face_prime = out['x_c_face']
                x_mouth_prime = out['x_c_mouth']
                dist_to_eye_surface = out['dist_eye']
                dist_to_face_surface = out['dist_face']
                dist_to_mouth_surface = out['dist_mouth']
                inside_bbox_eye = out['inside_bbox_eye']
                closest_vts_region = out['closest_vts_region']
                weights = out['weights']

            if self.dynamic_mask_basis is not None:
                pass # deprecated
            else:
                x_mask = None
                mask = None
                mask_region = None
                
            return {'canonical_eye':x_eye_prime, 'dist_to_eye_surface':dist_to_eye_surface, 'canonical_face':x_face_prime, 'dist_to_face_surface': dist_to_face_surface,\
                 'canonical_mouth':x_mouth_prime, 'dist_to_mouth_surface': dist_to_mouth_surface, 'inside_bbox_eye': inside_bbox_eye, 'closest_vts_region':closest_vts_region,\
                    'weights':weights, 'dynamic_mask':x_mask,'vts_mask':mask,'vts_mask_region':mask_region}
        # use single deformation field
        else:
            with torch.no_grad():
                out = self.flame_deform(x_t, shape_params,exp_params,pose_params,eye_pose_params, use_rotation_limits=use_rotation_limits, part=False)
                x_prime = out['x_c']
                dist_to_surface = out['dist']
                closest_vts_region = out['closest_vts_region']
                weights = out['weights']
                
            if self.dynamic_mask_basis is not None:
                pass # deprecated
            else:
                x_mask = None
                mask = None
                mask_region = None

            delta_x = torch.zeros_like(x_prime)
            x_c = x_prime + delta_x
            return {'canonical':x_c,'offset':delta_x,'dist_to_surface':dist_to_surface,'dynamic_mask':x_mask,'vts_mask':mask,'vts_mask_region':mask_region}

# Deformation module for Portrait4D (only head pose rotation)
@persistence.persistent_class
class DeformationModuleOnlyHead(torch.nn.Module):
    def __init__(self,
        flame_cfg_path = 'models/FLAME/cfg.yaml',
        flame_full = True,
        box_warp = 1.2
    ):
        super().__init__()
        self.flame_deform = FlameDeformationModuleOnlyHead(flame_cfg_path,flame_full=flame_full,box_warp=box_warp)

    def renderer(self, shape_params,exp_params,pose_params,eye_pose_params,camera, eye_blink_params=None, only_face=True,small_face=False,cull_backfaces=True,half_size=32,eye_mask=False,face_woeye=False,noinmouth=False,custom_feature=None,use_rotation_limits=False):
        # return Bx3xHxW correspondence map
        return self.flame_deform.renderer(shape_params,exp_params,pose_params,eye_pose_params,camera, eye_blink_params=eye_blink_params,only_face=only_face,small_face=small_face,cull_backfaces=cull_backfaces,half_size=half_size,eye_mask=eye_mask,face_woeye=face_woeye,noinmouth=noinmouth,custom_feature=custom_feature,use_rotation_limits=use_rotation_limits)
    
    def forward(self, x_t, shape_params,exp_params,pose_params,eye_pose_params, use_rotation_limits=False, smooth_th=3e-3):
        with torch.no_grad():
            out = self.flame_deform(x_t, shape_params,exp_params,pose_params,eye_pose_params, use_rotation_limits=use_rotation_limits, smooth_th=smooth_th)
            x_c = out['x_c']

        return {'canonical':x_c}