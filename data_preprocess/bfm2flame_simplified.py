import os
import click
from tqdm import tqdm
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from bfm2flame_mapper import Mapper

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from portrait4d.models.FLAME.lbs import batch_rodrigues

# rotation vector to matrix
def pose2rot(pose):
    rot_mats = batch_rodrigues(
        pose.view(-1, 3), dtype=pose.dtype).view([pose.shape[0], 3, 3])
    
    return rot_mats

# calculate camera intrinsics
def get_intrinsics(image_size=(512, 512), fov=12):

    K = np.zeros((3, 3)).astype(np.float32)

    K[0, 0] = 1 / (2 * np.tan(fov / 2 / 180 * np.pi))
    K[0, 2] = 1 / 2
    K[1, 1] = 1 / (2 * np.tan(fov / 2 / 180 * np.pi))
    K[1, 2] = 1 / 2
    K[2, 2] = 1

    return K


# transfer bfm three euler angles to rotation matrix
def bfm_angles_to_rotation(angles):
    # angles: [B, 3], three axis angles (x,y,z)
    n = angles.shape[0]
    phi = angles[:, 0]
    theta = angles[:, 1]
    roll = angles[:, 2]

    # Defined by Deep3DFaceRecon, rotation matrix of human head
    # In Deep3DFaceRecon, pos_rot = pos@rot^T = pos@(rot_x^T@rot_y^T@rot_z^T)

    rot_x = np.tile(np.expand_dims(np.eye(3),0),[n,1,1])
    rot_x[:,1,1] = np.cos(phi)
    rot_x[:,1,2] = -np.sin(phi)
    rot_x[:,2,1] = np.sin(phi)
    rot_x[:,2,2] = np.cos(phi)

    # rot_x = [ 1,   0,         0 ]
    #         [ 0, cos(x), -sin(x)]
    #         [ 0, sin(x),  cos(x)]

    rot_y = np.tile(np.expand_dims(np.eye(3),0),[n,1,1])
    rot_y[:,0,0] = np.cos(theta)
    rot_y[:,0,2] = np.sin(theta)
    rot_y[:,2,0] = -np.sin(theta)
    rot_y[:,2,2] = np.cos(theta) 

    # rot_y = [ cos(y), 0,  sin(y)]
    #         [ 0,      1,      0 ]
    #         [ -sin(y),0,  cos(y)]

    rot_z = np.tile(np.expand_dims(np.eye(3),0),[n,1,1])
    rot_z[:,0,0] = np.cos(roll)
    rot_z[:,0,1] = -np.sin(roll)
    rot_z[:,1,0] = np.sin(roll)
    rot_z[:,1,1] = np.cos(roll)

    # rot_z = [ cos(z),-sin(z), 0 ]
    #         [ sin(z), cos(z), 0 ]
    #         [ 0,      0,      1 ]

    # rot = rot_z @ rot_y @ rot_x
    rot = rot_z @ rot_y @ rot_x

    return rot

# bfm angles to flame rotation vectors (without align the canonical pose)
def bfm_angles_to_flame_poses(angles):
    rots = bfm_angles_to_rotation(angles)
    rots_matrix = R.from_matrix(rots)
    flame_poses = rots_matrix.as_rotvec()

    return flame_poses

# bfm angles to flame rotation vectors (after align the canonical pose)
def bfm_angles_to_flame_poses_rectified(angles, diff_euler=np.array([0.08489475, 0.002882, 0.00353599])): # calculate via first 1000 instances of FFHQ using calculate_diff_poses
    bfm2flame_poses = bfm_angles_to_flame_poses(angles)
    bfm2flame_rots = R.from_rotvec(bfm2flame_poses).as_matrix()

    diff_rot = R.from_euler('xyz',diff_euler).as_matrix()
    bfm2flame_rots_rectified  = diff_rot @ bfm2flame_rots
    bfm2flame_poses_rectified = R.from_matrix(bfm2flame_rots_rectified).as_rotvec()

    return bfm2flame_poses_rectified

# main function for bfm to flame parameter transfer
@click.command()
@click.option('--input_dir', type=str, help='Input folder', default='')
@click.option('--save_dir', type=str, help='Input folder', default='')
@click.option('--use_mapper', type=bool, help='Use bfm2flame mapper to obtain flame parameters', default=True)

def main(input_dir:str, save_dir:str, use_mapper:bool):
    
    root_path = input_dir
    save_path = os.path.join(save_dir,'bfm2flame_params_simplified')
    os.makedirs(save_path,exist_ok=True)
    all_files = sorted(os.listdir(os.path.join(root_path,'bfm_params')))
    
    if use_mapper:
        mapper = Mapper(in_dim=144, hidden_dim=512, out_dim=403, layers=2)
        mapper.load_state_dict(torch.load(os.path.join('assets','bfm2flame_mapper','model-iter200000.pth')))
        mapper = mapper.eval().cuda()
    
    for idx, file in tqdm(enumerate(all_files)):
        bfm_params = np.load(os.path.join(root_path,'bfm_params',file), allow_pickle=True).item()
        bfm_angles = bfm_params['angle']
        bfm2flame_poses = bfm_angles_to_flame_poses_rectified(bfm_angles).reshape(-1)
        
        if use_mapper:
            bfm_shapes = bfm_params['id']
            bfm_exps = bfm_params['exp']
            bfm_input = np.concatenate([bfm_shapes, bfm_exps], axis=-1).reshape(1,-1)
            with torch.no_grad():
                bfm_input = torch.from_numpy(bfm_input).cuda()
                bfm2flame_output = mapper(bfm_input).cpu().numpy()
            bfm2flame_shapes = bfm2flame_output[:,:300].reshape(-1)
            bfm2flame_exps = bfm2flame_output[:,300:400].reshape(-1)
            bfm2flame_jawposes = bfm2flame_output[:,400:].reshape(-1)
        else:
            bfm2flame_shapes = np.zeros([300])
            bfm2flame_exps = np.zeros([100])
            bfm2flame_jawposes = np.zeros([3])
        bfm2flame_eyeposes = np.zeros([6])
        
        c = np.zeros([25])
        intrinsics = get_intrinsics().reshape(-1)
        c[16:] = intrinsics
        extrinsics = np.eye(4).astype(np.float32)
        extrinsics[1,3] = 0.01
        extrinsics[2,3] = 4.2
        extrinsics = torch.from_numpy(extrinsics)
        rot_mat = pose2rot(-torch.from_numpy(bfm2flame_poses.astype(np.float32)).reshape(1,-1))  
        extrinsics[:3,:3] = rot_mat[0].float()
        extrinsics[:3,3] = (rot_mat[0].float() @ extrinsics[:3,3])[None]
        c[:16] = extrinsics.reshape(-1).numpy()
        
        bfm2flame_params = np.concatenate([c, bfm2flame_shapes, bfm2flame_exps, bfm2flame_poses, bfm2flame_jawposes, bfm2flame_eyeposes], axis=0).astype(np.float32)
        np.save(os.path.join(save_path,file), bfm2flame_params.reshape(-1))


if __name__ == '__main__':
    main()