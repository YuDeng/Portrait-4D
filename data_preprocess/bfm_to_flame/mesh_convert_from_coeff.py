'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''

import os
# import cv2
# import h5py
import numpy as np
import chumpy as ch
from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model
from scipy.io import loadmat
import click
from time import time

class ParametricFaceModel:
    def __init__(self, 
                bfm_folder='../assets/facerecon/bfm', 
                recenter=True,
                camera_distance=10.,
                init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                focal=1015.,
                center=112.,
                is_train=True,
                default_name='BFM_model.mat'):
        
        if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            transferBFM09(bfm_folder)
        model = loadmat(os.path.join(bfm_folder, default_name))
        # mean face shape. [3*N,1]
        self.mean_shape = model['meanshape'].astype(np.float32)
        # identity basis. [3*N,80]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3*N,64]
        self.exp_base = model['exBase'].astype(np.float32)
        # mean face texture. [3*N,1] (0-255)
        self.mean_tex = model['meantex'].astype(np.float32)
        # texture basis. [3*N,80]
        self.tex_base = model['texBase'].astype(np.float32)
        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1
        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1
        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        if is_train:
            # vertex indices for small face region to compute photometric error. starts from 0.
            self.front_mask = np.squeeze(model['frontmask2_idx']).astype(np.int64) - 1
            # vertex indices for each face from small face region. starts from 0. [f,3]
            self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
            # vertex indices for pre-defined skin region to compute reflectance loss
            self.skin_mask = np.squeeze(model['skinmask'])
        
        if recenter:
            shape_center = np.array([[-0.00322627, 0.04506068, 0.75983167]]).astype(np.float32)
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - shape_center
            self.mean_shape = mean_shape.reshape([-1, 1])

    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = np.einsum('ij,aj->ai', self.id_base, id_coeff)
        exp_part = np.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = np.ones([batch_size, 1])
        zeros = np.zeros([batch_size, 1])
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = np.concatenate([
            ones, zeros, zeros,
            zeros, np.cos(x), -np.sin(x), 
            zeros, np.sin(x), np.cos(x)
        ], axis=1).reshape([batch_size, 3, 3])
        
        rot_y = np.concatenate([
            np.cos(y), zeros, np.sin(y),
            zeros, ones, zeros,
            -np.sin(y), zeros, np.cos(y)
        ], axis=1).reshape([batch_size, 3, 3])

        rot_z = np.concatenate([
            np.cos(z), -np.sin(z), zeros,
            np.sin(z), np.cos(z), zeros,
            zeros, zeros, ones
        ], axis=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.transpose(0, 2, 1)

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + np.expand_dims(trans, 1)

    def compute_shape_transform(self, coef_dict):
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.compute_rotation(coef_dict['angle'])
        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])

        return face_shape_transformed


def convert_mesh(mesh, corr_setup):
    v = np.vstack((mesh.v, np.zeros_like(mesh.v)))
    return Mesh(corr_setup['mtx'].dot(v), corr_setup['f_out'])

def convert_BFM_mesh_to_FLAME(FLAME_model_fname, BFM_mesh_fname, FLAME_out_fname, BFM_model):
    '''
    Convert Basel Face Model mesh to a FLAME mesh
    \param FLAME_model_fname        path of the FLAME model
    \param BFM_mesh_fname           path of the BFM mesh to be converted
    \param FLAME_out_fname          path of the output file
    '''

    # Regularizer weights for jaw pose (i.e. opening of mouth), shape, and facial expression.
    # Increase regularization in case of implausible output meshes. 
    w_pose = 1e-4
    w_shape = 1e-3
    w_exp = 1e-4

    if not os.path.exists(os.path.dirname(FLAME_out_fname)):
        os.makedirs(os.path.dirname(FLAME_out_fname))

    if not os.path.exists(BFM_mesh_fname):
        print('BFM mesh not found %s' % BFM_mesh_fname)
        return
    BFM_coeff = np.load(BFM_mesh_fname, allow_pickle=True).item()
    BFM_v = BFM_model.compute_shape_transform(BFM_coeff).squeeze(0)
    BFM_f = BFM_model.face_buf
    BFM_mesh = Mesh(v=BFM_v,f=BFM_f)

    # BFM_mesh = Mesh(filename=BFM_mesh_fname)

    if not os.path.exists(FLAME_model_fname):
        print('FLAME model not found %s' % FLAME_model_fname)
        return
    model = load_model(FLAME_model_fname)

    if not os.path.exists('./data/BFM_to_FLAME_corr.npz'):
        print('Cached mapping not found')
        return
    cached_data = np.load('./data/BFM_to_FLAME_corr.npz', allow_pickle=True, encoding="latin1")

    BFM2017_corr = cached_data['BFM2017_corr'].item()
    BFM2009_corr = cached_data['BFM2009_corr'].item()
    BFM2009_cropped_corr = cached_data['BFM2009_cropped_corr'].item()

    print('input:',BFM_mesh.v.shape[0])
    print(BFM2009_cropped_corr['mtx'].shape[1])

    if (2*BFM_mesh.v.shape[0] == BFM2017_corr['mtx'].shape[1]) and (BFM_mesh.f.shape[0] == BFM2017_corr['f_in'].shape[0]):
        conv_mesh = convert_mesh(BFM_mesh, BFM2017_corr)
    elif (2*BFM_mesh.v.shape[0] == BFM2009_corr['mtx'].shape[1]) and (BFM_mesh.f.shape[0] == BFM2009_corr['f_in'].shape[0]):
        conv_mesh = convert_mesh(BFM_mesh, BFM2009_corr)
    elif (2*BFM_mesh.v.shape[0] == BFM2009_cropped_corr['mtx'].shape[1]) and (BFM_mesh.f.shape[0] == BFM2009_cropped_corr['f_in'].shape[0]):
        conv_mesh = convert_mesh(BFM_mesh, BFM2009_cropped_corr)
    else:
        print('Conversion failed - input mesh does not match any setup')
        return

    FLAME_mask_ids = cached_data['FLAME_mask_ids']

    scale = ch.ones(1)
    v_target = scale*ch.array(conv_mesh.v)
    dist = v_target[FLAME_mask_ids]-model[FLAME_mask_ids]
    pose_reg = model.pose[3:]
    shape_reg = model.betas[:300]
    exp_reg = model.betas[300:]
    obj = {'dist': dist, 'pose_reg': w_pose*pose_reg, 'shape_reg': w_shape*shape_reg, 'exp_reg': w_exp*exp_reg}
    ch.minimize(obj, x0=[scale, model.trans, model.pose[:3]])
    ch.minimize(obj, x0=[scale, model.trans, model.pose[np.hstack((np.arange(3), np.arange(6,9)))], model.betas])

    v_out = model.r/scale.r
    # Mesh(v_out, model.f).write_obj(FLAME_out_fname)
    params = {'pose':model.pose[:3],'trans':model.trans,'jaw':model.pose[6:9],'shape':model.betas[:300],'exp':model.betas[300:]}
    np.save(FLAME_out_fname,params)


@click.command()
@click.option('--input_dir', type=str, help='input bfm folder', default='/cpfs/shared/research-cv/3dgan/agent_test')
@click.option('--save_dir', type=str, help='input bfm folder', default='/cpfs/shared/research-cv/3dgan/agent_test')
@click.option('--group', type=int, help='input bfm folder', default=0)
@click.option('--num', type=int, help='input bfm folder', default=-1)
def main(
    input_dir:str, save_dir:str, group:int, num: int):
    # FLAME model filename (download from flame.is.tue.mpg.de)
    FLAME_model_fname = './model/generic_model.pkl'
    BFM_model = ParametricFaceModel(is_train=False)
    save_dir = os.path.join(save_dir,'bfm2flame_params')
    os.makedirs(save_dir,exist_ok=True)
    # BFM mesh to be converted 
    all_files = [f for f in os.listdir(os.path.join(input_dir,'bfm_params')) if f.endswith('.npy')]
    all_files = sorted(all_files)
    
    if num == -1:
        start = 0
        end = len(all_files)
    else:
        start = group*num
        end = (group+1)*num        
    for f in all_files[start:end]:
        print(f)
        start = time()
        BFM_mesh_fname = os.path.join(input_dir,'bfm_params',f)
        FLAME_out_fname = os.path.join(save_dir,f)
        convert_BFM_mesh_to_FLAME(FLAME_model_fname, BFM_mesh_fname, FLAME_out_fname, BFM_model)
        end = time()
        print('time:',end-start)

if __name__ == '__main__':
    print('Conversion started......')
    main()
    print('Conversion finished')