import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../portrait4d/')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.io import load_obj,save_obj
from portrait4d.training.deformer.deformation import FlameDeformationModule
from portrait4d.training.deformer.deform_utils import cam_world_matrix_transform
from torchvision.utils import save_image
import cv2
import time
import argparse

def ortho2persp(rotMat, image_size=(512, 512), fov=12):

    bs = rotMat.shape[0]
    dtype = rotMat.dtype
    device = rotMat.device

    K = torch.zeros((bs, 3, 3), dtype=dtype, device=device)
    RT = torch.zeros((bs, 3, 4), dtype=dtype, device=device)

    RT[:, [0, 1, 2], [0, 1, 2]] = 1
    RT[:, :3, :3] = rotMat


    RT[:, 2, 3] += -1 / (torch.tan(torch.Tensor([fov / 2 / 180 * np.pi])).float().cuda() * 6.2189)
    RT[:, 2, 3] *= 3 # match the scale of FlameDeformationModule

    K[:, 0, 0] = image_size[0] / (2 * torch.tan(torch.Tensor([fov / 2 / 180 * np.pi]))).float().cuda()
    K[:, 0, 2] = image_size[0] / 2
    K[:, 1, 1] = image_size[0] / (2 * torch.tan(torch.Tensor([fov / 2 / 180 * np.pi]))).float().cuda()
    K[:, 1, 2] = image_size[0] / 2
    K[:, 2, 2] = 1

    return K, RT

def get_eye_dist_dict(proj_pred_lmks70, gt, EYE_CLOSE_THRES=8):
    eye_dist_dict = dict()
    # compute flame projected eye dist
    EYE_UP_IDX = [36, 37, 38, 39, 42, 43, 44, 45]
    EYE_BOTTOM_IDX = [36, 41, 40, 39, 42, 47, 46, 45]
    eye_dist_dict['eye_up'] = eye_up = proj_pred_lmks70[:, EYE_UP_IDX, :]
    eye_dist_dict['eye_bottom'] = eye_bottom = proj_pred_lmks70[:, EYE_BOTTOM_IDX, :]
    eye_dist_dict['pred_dist'] = torch.sqrt(((eye_up - eye_bottom) ** 2).sum(2) + 1e-7)  # [bz, 4] # sq of diff-> sum -> sqrt => abs diff of corresponding points
    eye_dist_dict['righteye_pred_dist'] = torch.sqrt(((eye_up[:, :4, :] - eye_bottom[:, :4, :]) ** 2).sum(2) + 1e-7)
    eye_dist_dict['lefteye_pred_dist'] = torch.sqrt(((eye_up[:, 4:, :] - eye_bottom[:, 4:, :]) ** 2).sum(2) + 1e-7)

    # compute 2d lmk eye dist
    eye_dist_dict['eye_up_gt'] = eye_up_gt= gt[:, EYE_UP_IDX, :]
    eye_dist_dict['eye_bottom_gt'] = eye_bottom_gt = gt[:, EYE_BOTTOM_IDX, :]
    gt_dists = torch.sqrt(((eye_up_gt - eye_bottom_gt) ** 2).sum(2) + 1e-7)  # [bz, 4] # sq of diff-> sum -> sqrt => abs diff of corresponding points

    # force eye close
    gt_dists[gt_dists <= EYE_CLOSE_THRES] = 0
    eye_dist_dict['gt_dists'] = gt_dists

    return eye_dist_dict

def get_mouth_dist_dict(proj_pred_lmks70, gt):
    mouth_dist_dict = dict()
    MOUTH_UP_OUTER_IDX = [48, 49, 50, 51, 52, 53, 54]
    MOUTH_BOTTOM_OUTER_IDX = [48, 59, 58, 57, 56, 55, 54]
    MOUTH_UP_INNER_IDX = [60, 62, 64]
    MOUTH_BOTTOM_INNER_IDX = [60, 66, 64]
    # compute flame projected outer mouth dist
    mouth_dist_dict['mouth_up_outer'] = proj_pred_lmks70[:, MOUTH_UP_OUTER_IDX, :]
    mouth_dist_dict['mouth_bottom_outer'] = proj_pred_lmks70[:, MOUTH_BOTTOM_OUTER_IDX, :]
    # vpred_dist_mouth_outer = torch.sqrt(((mouth_up_outer - mouth_bottom_outer) ** 2).sum(2) + 1e-7)  # [bz, 4] # sq of diff-> sum -> sqrt => abs diff of corresponding points

    # # compute 2d lmk outer mouth dist
    mouth_dist_dict['mouth_up_gt_outer'] = gt[:, MOUTH_UP_OUTER_IDX, :]
    mouth_dist_dict['mouth_bottom_gt_outer'] = gt[:, MOUTH_BOTTOM_OUTER_IDX, :]
    # vgt_dists_mouth_outer = torch.sqrt(((mouth_up_gt_outer - mouth_bottom_gt_outer) ** 2).sum(2) + 1e-7)  # [bz, 4] # sq of diff-> sum -> sqrt => abs diff of corresponding points

    # # compute flame projected outer mouth dist
    mouth_dist_dict['mouth_up_inner'] = proj_pred_lmks70[:, MOUTH_UP_INNER_IDX, :]
    mouth_dist_dict['mouth_bottom_inner'] = proj_pred_lmks70[:, MOUTH_BOTTOM_INNER_IDX, :]
    # vpred_dist_mouth_inner = torch.sqrt(((mouth_up_inner - mouth_bottom_inner) ** 2).sum(2) + 1e-7)  # [bz, 4] # sq of diff-> sum -> sqrt => abs diff of corresponding points

    # # compute 2d lmk outer mouth dist
    mouth_dist_dict['mouth_up_gt_inner'] = gt[:, MOUTH_UP_INNER_IDX, :]
    mouth_dist_dict['mouth_bottom_gt_inner'] = gt[:, MOUTH_BOTTOM_INNER_IDX, :]
    
    return mouth_dist_dict


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='')
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--batchsize", type=int, default=100)
parser.add_argument("--visualize", default=False, action='store_true')
args = parser.parse_args()

def optimize():
    
    # define FLAME model
    flamedeform = FlameDeformationModule('FLAME/cfg.yaml',flame_full=True).cuda()
    
    # vertex indices without eye region
    wo_eyelids_idx = [i for i in range(5023) if i not in flamedeform.eyelids_idxs]
    
    # transfer 98 landmarks to 70 landmarks to match FLAME
    idx_98to70 = [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20,
                22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37,
                42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56,
                57, 58, 59, 60, 61, 63, 64, 65, 67, 68, 69,
                71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82,
                83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                94, 95, 96, 97 ]

    eye_in_shape = [2422,2422, 2452, 2454, 2471, 3638, 2276, 2360, 3835, 1292, 1217, 1146, 1146, 999, 827, ]
    eye_in_shape_reduce = [0,2,4,5,6,7,8,9,10,11,13,14]
    # right [(2420,2422), (2452, 2454), 2471, 3638, 2276, 2360]
    # left [3835, 1292, 1217, (1143, 1146), 999, 827, ]
    # Up [(2452 2454), 2471] [1292, 1217]
    # Bottom [2360, 2276] [829, 999]
    
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir,exist_ok=True)
    
    # load data
    data_list = sorted([os.path.join(save_dir,'align_images',f) for f in os.listdir(os.path.join(save_dir,'align_images')) if f.endswith('png') or f.endswith('jpg')])
    param_list = [f.replace('align_images','bfm2flame_params').replace('.jpg','.npy').replace('.png','.npy') for f in data_list]
    
    ldmks_2d_list = []
    ldmks_3d_list = []
    
    # load landmarks
    for idx, img_path in enumerate(data_list):
        
        img_name = img_path.split('/')[-1].replace('.jpg','').replace('.png','')
        ldmks_2d_path = os.path.join(data_dir, '2dldmks_align', img_name+'.npy')
        ldmks_3d_path = os.path.join(data_dir, '3dldmks_align', img_name+'.npy')

        ldmks_2d = np.load(ldmks_2d_path)
        ldmks_3d = np.load(ldmks_3d_path)
        ldmks_2d_list.append(ldmks_2d)
        ldmks_3d_list.append(ldmks_3d)

    ldmks_2d_list = np.stack(ldmks_2d_list,0)
    ldmks_3d_list = np.stack(ldmks_3d_list,0)
    ldmks_3d = ldmks_3d_list
    ldmks_3d = ldmks_3d[...,:2].reshape(-1,68,2)
    ldmks_2d = ldmks_2d_list
    ldmks_2d = ldmks_2d.reshape(-1,98,2)
    
    # hyper-parameters for optimization
    batchsize = args.batchsize
    flen = len(data_list)
    n_iter_sec1 = 1500
    n_iter_sec2 = 3000
    image_size = (512, 512)
    
    # optimize in parallel
    for i in range(0, flen, batchsize):
        print(i)
        
        # load initial FLAME parameters of current batch
        cur_shape_params = []
        cur_exp_params = []
        cur_pose_params = []
        for k in range(min(batchsize,flen-i)):
            params_bfm2flame = np.load(param_list[i+k], allow_pickle=True).item()
            cur_shape_param = torch.from_numpy(np.array(params_bfm2flame['shape'][None, :300])).to(torch.float32).cuda()
            cur_exp_param = torch.from_numpy(np.array(params_bfm2flame['exp'][None, :100])).to(torch.float32).cuda()
            cur_pose_param = torch.from_numpy(np.concatenate([params_bfm2flame['pose'][None, :],params_bfm2flame['jaw'][None, :]],axis=1)).to(torch.float32).cuda()
            cur_shape_params.append(cur_shape_param)
            cur_exp_params.append(cur_exp_param)
            cur_pose_params.append(cur_pose_param)
        
        cur_shape_params = torch.cat(cur_shape_params,dim=0)
        cur_exp_params = torch.cat(cur_exp_params,dim=0)
        cur_pose_params = torch.cat(cur_pose_params,dim=0)
        
        init_shape_params = cur_shape_params.clone()
        init_exp_params = cur_exp_params.clone()
        eye_pose_params = torch.zeros_like(cur_pose_params)
        
        # obtain gt landmarks for optimization
        ldmk_gt = torch.from_numpy(ldmks_2d[i:min(i+batchsize, flen), idx_98to70]).float().cuda()
        ldmk_gt[..., 1] = 512 - ldmk_gt[..., 1]
        ldmk_gt_3d = torch.from_numpy(ldmks_3d[i:min(i+batchsize, flen)]).float().cuda()
        ldmk_gt_3d[..., 1] = 512 - ldmk_gt_3d[..., 1]
        gt = torch.cat([ldmk_gt_3d[:, :17, :2], ldmk_gt[:, 17:, :2]], 1)

        # initialize camera translation
        t = torch.zeros(1, 3).float().cuda().repeat(cur_shape_params.shape[0], 1)
        t[:, 2] = -1 * 3 / (torch.tan(torch.Tensor([12 / 2 / 180 * np.pi])).float().cuda() * 6.2189)
        
        # define learnable variables
        translate = torch.nn.Parameter(t, requires_grad=True)
        optim_shape_params = torch.nn.Parameter(cur_shape_params.float().cuda(), requires_grad=True)
        optim_exp_params = torch.nn.Parameter(cur_exp_params.float().cuda(), requires_grad=True)
        optim_jaw_pose_params = torch.nn.Parameter(cur_pose_params[:,3:6].float().cuda(), requires_grad=True)
        optim_eye_pose_params = torch.nn.Parameter(torch.zeros_like(cur_pose_params[:, :3]).float().cuda(), requires_grad=True)
        optim_pose_params = torch.nn.Parameter(cur_pose_params[:, :3].float().cuda(), requires_grad=True)
        
        # Only optimize camera translation in stage 1
        params = [ {"params": [translate]} ]
        
        # define optimizer and loss
        optimizer = optim.Adam(params, lr=0.001)
        optimizer.zero_grad()
        loss_fn = nn.MSELoss()
        min_loss = 1000000
        cur_shape_min = None
        cur_exp_min = None
        cur_jawpose_min = None
        cur_eyepose_min = None
        cur_translate_min = None
        w2c_matrix_min = None
        cur_3ds_min = None

        start_time = time.time()
        print("Stage 1, optimizing translation...")
        
        ################### Stage 1 for translation optimization
        
        for ii in range(n_iter_sec1):
            
            # calculate world to camera transformation
            world2cam_rot = flamedeform.flame_model._pose2rot(optim_pose_params)
            K, RT = ortho2persp(world2cam_rot, image_size=image_size)
            world2cam_matrix = torch.eye(4).unsqueeze(0).repeat(world2cam_rot.shape[0],1,1).cuda()
            w2c_matrix_temp = torch.cat([RT[:, :3, :3], translate[..., None]], 2)
            w2c_matrix = torch.cat([w2c_matrix_temp, world2cam_matrix[:, 3:, :]], 1)
            
            # obtain reconstructed FLAME mesh
            cur_shape, _, cur_3ds, joints = flamedeform.flame_model(optim_shape_params, optim_exp_params, torch.cat([torch.zeros_like(cur_pose_params[:,:3]), optim_jaw_pose_params], 1), torch.cat([optim_eye_pose_params, optim_eye_pose_params], 1))
            cur_eye = cur_shape[:, eye_in_shape]
            cur_eye[:, 0] = (cur_eye[:, 0] + cur_eye[:, 1]) * 0.5
            cur_eye[:, 2] = (cur_eye[:, 2] + cur_eye[:, 3]) * 0.5
            cur_eye[:, 11] = (cur_eye[:, 11] + cur_eye[:, 12]) * 0.5
            cur_eye = cur_eye[:, eye_in_shape_reduce]
            cur_3ds[:, [37,38,40,41,43,44,46,47]] = cur_eye[:, [1,2,4,5,7,8,10,11]]
            cur_3ds_ = torch.cat([cur_3ds, torch.ones(cur_3ds.shape[0], cur_3ds.shape[1], 1).cuda()], 2)

            cur_3ds = torch.bmm(w2c_matrix, cur_3ds_.permute(0,2,1)).permute(0,2,1) # (B,N,4)
            cur_3ds = cur_3ds[...,:3]
            cur_3ds[..., -1] = -cur_3ds[..., -1] # reverse z axis to fit pytorch3d renderer
            cur_3ds = torch.bmm(K, cur_3ds.permute(0, 2, 1)).permute(0, 2, 1)
            cur_3ds /= cur_3ds[..., -1:].clone()
            proj_pred_lmks70 = cur_3ds[:, :, :2]

            eye_dist_dict = get_eye_dist_dict(proj_pred_lmks70, gt)
            mouth_dist_dict = get_mouth_dist_dict(proj_pred_lmks70, gt)
            
            # calculate loss
            loss = loss_fn(proj_pred_lmks70, gt) + \
                2 * loss_fn(proj_pred_lmks70[:, [36, 39, 42, 45]], gt[:, [36, 39, 42, 45]]) + \
                2 * loss_fn(proj_pred_lmks70[:, 48:68], gt[:, 48:68])

            loss += loss_fn(eye_dist_dict['eye_up'], eye_dist_dict['eye_up_gt']) +  loss_fn(eye_dist_dict['eye_bottom'], eye_dist_dict['eye_bottom_gt'])

            if loss.item() < min_loss:
                min_loss = loss.item()
                w2c_matrix_min = w2c_matrix.clone().detach()
                cur_shape_min = optim_shape_params.clone().detach()
                cur_exp_min = optim_exp_params.clone().detach()
                cur_jawpose_min = optim_jaw_pose_params.clone().detach()
                cur_eyepose_min = optim_eye_pose_params.clone().detach()
                cur_translate_min = translate.clone().detach()
                pose_min = optim_pose_params.clone().detach()

            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()


        ################### Stage 2 for shape, expression, eye, and translation optimization

        optim_shape_params = torch.nn.Parameter(cur_shape_min.float().cuda(), requires_grad=True)
        optim_exp_params = torch.nn.Parameter(cur_exp_min.float().cuda(), requires_grad=True)
        optim_eye_pose_params = torch.nn.Parameter(cur_eyepose_min.float().cuda(), requires_grad=True)
        optim_pose_params = torch.nn.Parameter(pose_min.float().cuda(), requires_grad=True)
        translate = torch.nn.Parameter(cur_translate_min.float().cuda(), requires_grad=True)
        optim_jaw_pose_params = torch.nn.Parameter(cur_jawpose_min.float().cuda(), requires_grad=True)
        
        params = [ {"params": [optim_shape_params, optim_exp_params, optim_eye_pose_params, translate]} ]
        optimizer = optim.Adam(params, lr=0.001)
        optimizer.zero_grad()
        loss_fn = nn.MSELoss()
        min_loss = 1000000
        cur_exp_min = None
        cur_eyepose_min = None
        
        print("Stage 2, optimizing shape, expression, eye, and translation...")
        for ii in range(n_iter_sec2):
            
            world2cam_rot = flamedeform.flame_model._pose2rot(optim_pose_params)
            K, RT = ortho2persp(world2cam_rot, image_size=image_size)
            world2cam_matrix = torch.eye(4).unsqueeze(0).repeat(world2cam_rot.shape[0],1,1).cuda()
            w2c_matrix_temp = torch.cat([RT[:, :3, :3], translate[..., None]], 2)
            w2c_matrix = torch.cat([w2c_matrix_temp, world2cam_matrix[:, 3:, :]], 1)

            cur_shape, _, cur_3ds, joints = flamedeform.flame_model(optim_shape_params, optim_exp_params, torch.cat([torch.zeros_like(cur_pose_params[:,:3]), optim_jaw_pose_params], 1), torch.cat([optim_eye_pose_params, optim_eye_pose_params], 1))
            init_shape, _, _, _ = flamedeform.flame_model(init_shape_params, init_exp_params, torch.cat([torch.zeros_like(cur_pose_params[:,:3]), optim_jaw_pose_params], 1), torch.cat([optim_eye_pose_params, optim_eye_pose_params], 1))
            
            cur_eye = cur_shape[:, eye_in_shape]
            cur_eye[:, 0] = (cur_eye[:, 0] + cur_eye[:, 1]) * 0.5
            cur_eye[:, 2] = (cur_eye[:, 2] + cur_eye[:, 3]) * 0.5
            cur_eye[:, 11] = (cur_eye[:, 11] + cur_eye[:, 12]) * 0.5
            cur_eye = cur_eye[:, eye_in_shape_reduce]
            cur_3ds[:, [37,38,40,41,43,44,46,47]] = cur_eye[:, [1,2,4,5,7,8,10,11]]
            cur_3ds_ = torch.cat([cur_3ds, torch.ones(cur_3ds.shape[0], cur_3ds.shape[1], 1).cuda()], 2)

            cur_3ds = torch.bmm(w2c_matrix, cur_3ds_.permute(0,2,1)).permute(0,2,1) # (B,N,4)
            cur_3ds = cur_3ds[...,:3]
            cur_3ds[..., -1] = -cur_3ds[..., -1] # reverse z axis to fit pytorch3d renderer
            cur_3ds = torch.bmm(K, cur_3ds.permute(0, 2, 1)).permute(0, 2, 1)
            cur_3ds /= cur_3ds[..., -1:].clone()
            proj_pred_lmks70 = cur_3ds[:, :, :2]

            eye_dist_dict = get_eye_dist_dict(proj_pred_lmks70, gt)
            mouth_dist_dict = get_mouth_dist_dict(proj_pred_lmks70, gt)
            
            # calculate loss
            loss = 5 * (loss_fn(eye_dist_dict['eye_up'], eye_dist_dict['eye_up_gt']) + loss_fn(eye_dist_dict['eye_bottom'], eye_dist_dict['eye_bottom_gt'])) + \
                    2 * loss_fn(eye_dist_dict['pred_dist'], eye_dist_dict['gt_dists']) + \
                    loss_fn(eye_dist_dict['righteye_pred_dist'], eye_dist_dict['lefteye_pred_dist']) + 1 * loss_fn(proj_pred_lmks70[:, :], gt[:, :]) + \
                    torch.sum(torch.abs(optim_exp_params-init_exp_params), 1).mean(0) * 0.5 + \
                    torch.sum(torch.abs(optim_shape_params-init_shape_params), 1).mean(0) * 4
            
            loss += loss_fn(init_shape[:,wo_eyelids_idx,:3].detach(),cur_shape[:,wo_eyelids_idx,:3])*1e6 + torch.sum(torch.abs(optim_eye_pose_params),1).mean(0) * 0.1

            if loss.item() < min_loss:
                min_loss = loss.item()
                w2c_matrix_min = w2c_matrix.clone().detach()
                cur_shape_min = optim_shape_params.clone().detach()
                cur_exp_min = optim_exp_params.clone().detach()
                cur_jawpose_min = optim_jaw_pose_params.clone().detach()
                cur_eyepose_min = optim_eye_pose_params.clone().detach()
                cur_translate_min = translate.clone().detach()
                pose_min = optim_pose_params.clone().detach()
                cur_3d_min = cur_3ds.clone().detach()
                cur_s_min = cur_shape.clone().detach()

            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()
            
        # finalize camera parameters
        cur_cam2world_matrix = cam_world_matrix_transform(w2c_matrix_min).clone().detach()
        cur_K = K.clone().detach()
        
        # normalize camera intrinsics
        cur_K[:, 0, 0] /= image_size[0]
        cur_K[:, 0, 2] /= image_size[0]
        cur_K[:, 1, 1] /= image_size[0]
        cur_K[:, 1, 2] /= image_size[0]
        
        c_compose = torch.cat([cur_cam2world_matrix.reshape(-1,16),cur_K.reshape(-1,9)],dim=1)
        
        # post-process eye gaze to prevent inward eye balls
        for q in range(len(cur_eyepose_min)):
            p = torch.norm(cur_eyepose_min[q], p=2, dim=0)
            if p > torch.pi/2:
                cur_eyepose_min[q] = cur_eyepose_min[q]/p*(p-torch.pi)
        
        # finalize optimized flame parameters
        cur_labels = torch.cat([c_compose,cur_shape_min.detach(),cur_exp_min,torch.cat([cur_pose_params[:,:3],cur_jawpose_min],dim=1),torch.cat([cur_eyepose_min, cur_eyepose_min], 1)],dim=-1).cpu().numpy()
        
        # for visualization only
        if args.visualize:
            with torch.no_grad():
                uv_compose = flamedeform.renderer(cur_shape_min.clone().detach(),cur_exp_min.clone().detach(),torch.cat([torch.zeros_like(cur_pose_params[:,:3]),cur_jawpose_min],dim=1),torch.cat([cur_eyepose_min, cur_eyepose_min], 1),c_compose, fov=12, half_size=256)[0]

                uv_head = []
                uv_mask = []
                for k in range(len(cur_shape_min)):
                    uv_head_, uv_mask_ = flamedeform.renderer_visualize(cur_shape_min[k:k+1].clone().detach(),cur_exp_min[k:k+1].clone().detach(),torch.cat([torch.zeros_like(cur_pose_params[k:k+1,:3]),cur_jawpose_min[k:k+1]],dim=1),torch.cat([cur_eyepose_min[k:k+1], cur_eyepose_min[k:k+1]], 1),c_compose[k:k+1], fov=12, half_size=256)
                    uv_head.append(uv_head_)
                    uv_mask.append(uv_mask_)
                uv_head = torch.cat(uv_head,dim=0)
                uv_mask = torch.cat(uv_mask,dim=0)

                _, face_wo_eye_mask = flamedeform.renderer_visualize(cur_shape_min.clone().detach(),cur_exp_min.clone().detach(),torch.cat([torch.zeros_like(cur_pose_params[:,:3]),cur_jawpose_min],dim=1),torch.cat([cur_eyepose_min, cur_eyepose_min], 1),c_compose, fov=12, half_size=256,face_woeye=True,eye_mask=False)
                _, eye_mask = flamedeform.renderer_visualize(cur_shape_min.clone().detach(),cur_exp_min.clone().detach(),torch.cat([torch.zeros_like(cur_pose_params[:,:3]),cur_jawpose_min],dim=1),torch.cat([cur_eyepose_min, cur_eyepose_min], 1),c_compose, fov=12, half_size=256,face_woeye=False,eye_mask=True)
                
                eye_mask = eye_mask * (1-face_wo_eye_mask)

        for j in range(len(cur_labels)):

            img_path = data_list[i+j]
            img_name = img_path.split("/")[-1].replace('.png','').replace('.jpg','')
            save_label_dir = os.path.join(save_dir, "flame_optim_params")
            os.makedirs(save_label_dir, exist_ok=True)
            np.save(os.path.join(save_label_dir,'{}.npy'.format(img_name)), cur_labels[j])
            
            if args.visualize:
                save_uv_dir = os.path.join(save_dir, "flame_optim_uvs")
                save_show_dir = os.path.join(save_dir, "flame_optim_visuals")
                os.makedirs(save_uv_dir, exist_ok=True)
                os.makedirs(save_show_dir, exist_ok=True)

                save_image(uv_compose[j], os.path.join(save_uv_dir, '{}.jpg'.format(img_name)), normalize=True, range=(-1, 1))
                save_image(uv_head[j], os.path.join(save_uv_dir, '{}_h.jpg'.format(img_name)), normalize=True, range=(-1, 1))

                img = cv2.imread(img_path)
                img = cv2.resize(img, (512, 512))
                uv = cv2.imread(os.path.join(save_uv_dir, '{}.jpg'.format(img_name)))
                mask = (uv == 127).astype(np.float32)
                img_s = img * mask + uv * (1 - mask) * 0.5 + img * (1 - mask) * 0.5
                uvh = cv2.imread(os.path.join(save_uv_dir, '{}_h.jpg'.format(img_name)))
                mask = (uvh == 127).astype(np.float32)
                img_h = img * mask + uvh * (1 - mask) * 0.7 + img * (1 - mask) * 0.3
                eye_mask_ =  eye_mask[j].permute(1,2,0).cpu().numpy().astype('uint8')
                eye_mask_ = np.concatenate([np.zeros_like(eye_mask_), eye_mask_, np.zeros_like(eye_mask_)], -1) * 255
                img_e = cv2.addWeighted(img, 1, eye_mask_, 0.5, 1)

                for k in range(cur_3d_min[j].shape[0]):
                    x = int(cur_3d_min[j][k][0].detach().cpu().numpy())
                    y = int(cur_3d_min[j][k][1].detach().cpu().numpy())
                    y = 512 - y
                    x = min(max(x,1),510)
                    y = min(max(y,1),510)
                    for n in range(-1, 2):
                        for m in range(-1, 2):
                            img_s[y+n, x+m] = [0, 0, 255]
                            img_h[y+n, x+m] = [0, 0, 255]
                            img_e[y+n, x+m] = [0, 0, 255]

                    xx = int(ldmks_2d[i+j, idx_98to70][k][0])
                    yy = int(ldmks_2d[i+j, idx_98to70][k][1])

                    for n in range(-1, 2):
                        for m in range(-1, 2):
                            img_s[yy+n, xx+m] = [255, 0, 0]
                            img_h[yy+n, xx+m] = [255, 0, 0]
                            img_e[yy+n, xx+m] = [255, 0, 0]

                img_s= np.concatenate([img_s, img_h, img_e], 1)    
                cv2.imwrite(os.path.join(save_show_dir, "{}.jpg".format(img_name)), (img_s).astype(np.uint8))

        end_time = time.time()
        print("batch time: {}".format((end_time-start_time) / (i+batchsize) * batchsize ))

    end_time = time.time()

if __name__ == '__main__':
    optimize()