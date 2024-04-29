# Main training loop of Portrait4D, modified from EG3D: https://github.com/NVlabs/eg3d

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import random
import torch
import torch.nn as nn
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from models.lpips.lpips import LPIPS
from models.arcface.iresnet import iresnet18
from models.pdfgc.encoder import FanEncoder
# from training.triplane import PartTriPlaneGeneratorDeform

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(all_shape_params, all_exp_params, all_pose_params, all_eye_pose_params, all_c, static_dataset=False, random_seed=1):
    gw = 7
    gh = 4

    grid_indices = np.random.RandomState(random_seed).randint(0,len(all_shape_params),size=(gw * gh))

    shape_params = all_shape_params[grid_indices]
    shape_params = np.tile(np.expand_dims(shape_params,1),(1,3,1)).reshape(gw * gh, 3, -1)

    grid_indices2 = np.random.RandomState(random_seed+1).randint(0,len(all_exp_params),size=(gw * gh))
    mot_indices = np.random.RandomState(random_seed+2).randint(0,len(all_exp_params[0]),size=(gw * gh, 2))

    exp_params = all_exp_params[grid_indices2]
    exp_params = np.stack([exp_params[i,mot_indices[i]] for i in range(len(mot_indices))]) # (gw * gh, 2, dim)

    pose_params = all_pose_params[grid_indices2]
    pose_params = np.stack([pose_params[i,mot_indices[i]] for i in range(len(mot_indices))]) # (gw * gh, 2, dim)

    eye_pose_params = all_eye_pose_params[grid_indices2]
    eye_pose_params = np.stack([eye_pose_params[i,mot_indices[i]] for i in range(len(mot_indices))]) # (gw * gh, 2, dim)

    if not static_dataset:
        # for dynamic
        exp_params = np.concatenate([exp_params, exp_params[:,-1:]], axis=1).reshape(gw * gh, 3, -1) # (gw * gh, 3, dim)
        pose_params = np.concatenate([pose_params, pose_params[:,-1:]], axis=1).reshape(gw * gh, 3, -1)
        eye_pose_params = np.concatenate([eye_pose_params, eye_pose_params[:,-1:]], axis=1).reshape(gw * gh, 3, -1)
    else:
        # for static
        exp_params = np.concatenate([exp_params[:,:1], exp_params[:,:1], exp_params[:,:1]], axis=1).reshape(gw * gh, 3, -1) # (gw * gh, 3, dim)
        pose_params = np.concatenate([pose_params[:,:1], pose_params[:,:1], pose_params[:,:1]], axis=1).reshape(gw * gh, 3, -1)
        eye_pose_params = np.concatenate([eye_pose_params[:,:1], eye_pose_params[:,:1], eye_pose_params[:,:1]], axis=1).reshape(gw * gh, 3, -1)


    grid_indices3 = np.random.randint(0,len(all_c),size=(gw * gh * 3))
    c = all_c[grid_indices3].reshape(gw * gh, 3, -1)

    return (gw, gh), shape_params, exp_params, pose_params, eye_pose_params, c

#----------------------------------------------------------------------------

def save_image_grid_all(img_app, img_mot, img_recon, fname, drange, grid_size):
    lo, hi = drange
    img_app = np.asarray(img_app, dtype=np.float32)
    img_app = (img_app - lo) * (255 / (hi - lo))
    img_app = np.rint(img_app).clip(0, 255).astype(np.uint8)

    img_mot = np.asarray(img_mot, dtype=np.float32)
    img_mot = (img_mot - lo) * (255 / (hi - lo))
    img_mot = np.rint(img_mot).clip(0, 255).astype(np.uint8)

    img_recon = np.asarray(img_recon, dtype=np.float32)
    img_recon = (img_recon - lo) * (255 / (hi - lo))
    img_recon = np.rint(img_recon).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img_app.shape

    img = np.concatenate([img_app,img_mot,img_recon],-1)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
#----------------------------------------------------------------------------

# choose random FLAME parameters for online data synthesis (torch version)
def fetch_random_params(all_shape_params, all_exp_params_static, all_pose_params_static, all_eye_pose_params_static, all_exp_params, all_pose_params, all_eye_pose_params, all_c, batchsize, static_dataset=False, half_static=False):
    # all_shape_params: [b1,300]
    # all_exp_params_static: [b1,100]
    # all_pose_params_static: [b1,6]
    # all_eye_pose_params_static: [b1,6]
    # all_c: [b1,25]
    # all_exp_params: [b2,M,100]
    # all_pose_params: [b2,M,6]
    # all_eye_pose_params: [b3,6]

    shape_index = torch.randint(0,len(all_shape_params),(batchsize,))
    c_index = torch.randint(0,len(all_c),(batchsize*3,))

    exp_index_static = torch.randint(0,len(all_exp_params_static),(batchsize//2,))
    eye_pose_index_static = torch.randint(0,len(all_exp_params_static),(batchsize//2,))

    exp_index_row = torch.randint(0,len(all_exp_params),(batchsize,))
    exp_index_col = torch.randint(0,len(all_exp_params[0]),(batchsize, 2))

    eye_pose_index = torch.randint(0,len(all_eye_pose_params),(batchsize, 2))

    cur_shape_params = all_shape_params[shape_index] # [b,300]
    cur_shape_params = cur_shape_params.unsqueeze(1).repeat(1,3,1) # [b,3,300]

    cur_c = all_c[c_index].reshape(batchsize, 3, -1) # [b,3,25]

    cur_exp_params_static = all_exp_params_static[exp_index_static] # [b/2,100]
    cur_exp_params_static = cur_exp_params_static.unsqueeze(1).repeat(1,3,1) # [b/2,3,100]

    cur_pose_params_static = all_pose_params_static[exp_index_static] # [b/2,6]
    cur_pose_params_static = cur_pose_params_static.unsqueeze(1).repeat(1,3,1) # [b/2,3,6]

    cur_eye_pose_params_static = all_eye_pose_params_static[eye_pose_index_static] # [b/2,6]
    cur_eye_pose_params_static = cur_eye_pose_params_static.unsqueeze(1).repeat(1,3,1) # [b/2,3,6]

    cur_exp_params = all_exp_params[exp_index_row]
    cur_exp_params = torch.stack([cur_exp_params[i,exp_index_col[i]] for i in range(len(exp_index_col))]) # [b,2,100]

    if not static_dataset:
        # for dynamic
        cur_exp_params = torch.cat([cur_exp_params,cur_exp_params[:,-1:]], dim=1) # [b,3,100]
    else:
        # for static
        cur_exp_params = torch.cat([cur_exp_params[:,:1],cur_exp_params[:,:1],cur_exp_params[:,:1]], dim=1) # [b,3,100]
    
    if half_static:
        cur_exp_params = torch.cat([cur_exp_params[:batchsize//2],cur_exp_params_static], dim=0)

    cur_pose_params = all_pose_params[exp_index_row]
    cur_pose_params = torch.stack([cur_pose_params[i,exp_index_col[i]] for i in range(len(exp_index_col))]) # [b,2,6]

    cur_pose_jaw_params = cur_pose_params[...,3:4]
    cur_pose_jaw_params_rand = torch.rand(*cur_pose_jaw_params.shape, device=cur_pose_jaw_params.device)*0.2
    jaw_prob = torch.rand(*cur_pose_jaw_params.shape, device=cur_pose_jaw_params.device)
    cur_pose_jaw_params = torch.where(jaw_prob < 0.5, cur_pose_jaw_params, cur_pose_jaw_params_rand)

    cur_pose_params = torch.cat([cur_pose_params[...,:3],cur_pose_jaw_params,cur_pose_params[...,4:]], dim=-1) # [b,2,6]

    if not static_dataset:
        # for dynamic
        cur_pose_params = torch.cat([cur_pose_params, cur_pose_params[:,-1:]], dim=1) # [b,3,6]
    else:
        # for static
        cur_pose_params = torch.cat([cur_pose_params[:,:1], cur_pose_params[:,:1], cur_pose_params[:,:1]], dim=1) # [b,3,6]
    
    if half_static:
        cur_pose_params = torch.cat([cur_pose_params[:batchsize//2], cur_pose_params_static], dim=0)

    cur_eye_pose_params = all_eye_pose_params[eye_pose_index] # [b,2,6]
    if not static_dataset:
        # for dynamic
        cur_eye_pose_params = torch.cat([cur_eye_pose_params, cur_eye_pose_params[:,-1:]], dim=1) # [b,3,6]
    else:
        # for static
        cur_eye_pose_params = torch.cat([cur_eye_pose_params[:,:1], cur_eye_pose_params[:,:1], cur_eye_pose_params[:,:1]], dim=1) # [b,3,6]
    
    if half_static:
        cur_eye_pose_params = torch.cat([cur_eye_pose_params[:batchsize//2], cur_eye_pose_params_static], dim=0)

    return cur_shape_params, cur_exp_params, cur_pose_params, cur_eye_pose_params, cur_c


#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    D_patch_kwargs          = {},       # Options for patch discriminator (deprecated).
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    D_patch_opt_kwargs      = {},       # Options for patch discriminator optimizer (deprecated).
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    D_patch_reg_interval    = 16,       # How often to perform regularization for D patch (deprecated)
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    motion_scale            = 1.0,      # Scale of the motion-related cross-attention outputs.
    swapping_prob           = 0.5,      # Probability to set dynamic data as static data.
    half_static             = False,    # Whether or not to set the second half of the batchsize as static data.
    resume_pkl_G_syn        = None,     # Checkpoint of pre-trained GenHead generator for training data synthesis.
    truncation_psi          = 0.7,      # Truncation rate of GenHead for training data synthesis.
    cross_lr_scale          = 1.0       # Learning rate scale of the motion-related cross attentions.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    batch_size_dataset = batch_size//num_gpus
    # all_shape_n_c_params = np.load('/cpfs/share/data/3dgan/lmdbs_v2/ffhq_all_shape_n_c_params.npy')   # [b, N, 325]
    all_shape_n_c_params = np.load(training_set_kwargs.shape_n_c_params_path1)   # [b, N, 325]
    all_shape_n_c_params = all_shape_n_c_params.reshape(-1,325)

    # all_shape_n_c_params2 = np.load('/cpfs/share/data/3dgan/lmdbs_v2/vfhq_all_shape_n_c_params.npy')
    all_shape_n_c_params2 = np.load(training_set_kwargs.shape_n_c_params_path2)
    all_shape_n_c_params2 = all_shape_n_c_params2.reshape(-1,325)
    all_shape_n_c_params = np.concatenate([all_shape_n_c_params, all_shape_n_c_params2], axis=0)

    all_shape_params = all_shape_n_c_params[:,25:]
    all_c = all_shape_n_c_params[:,:25]

    # all_motion_params_static = np.load('/cpfs/share/data/3dgan/lmdbs_v2/ffhq_all_motion_params.npy')   # [b, N, 112]
    all_motion_params_static = np.load(training_set_kwargs.motion_params_path1)   # [b, N, 112]
    all_motion_params_static = all_motion_params_static.reshape(-1,112)
    all_exp_params_static = all_motion_params_static[:,:100]
    all_pose_params_static = all_motion_params_static[:,100:106]
    all_pose_params_static = np.concatenate([0*all_pose_params_static[...,:3],all_pose_params_static[...,3:]],axis=-1)
    all_eye_pose_params_static = all_motion_params_static[:,106:]

    # all_motion_params = np.load('/cpfs/share/data/3dgan/lmdbs_v2/vfhq_all_motion_params.npy')  # [b, M, 112]
    all_motion_params = np.load(training_set_kwargs.motion_params_path2)  # [b, M, 112]
    all_exp_params = all_motion_params[:,:,:100]
    all_pose_params = all_motion_params[:,:,100:106]
    all_pose_params = np.concatenate([0*all_pose_params[...,:3],all_pose_params[...,3:]],axis=-1)
    all_eye_pose_params = all_motion_params[:,:,106:]


    if rank == 0:
        print('Loading training set...')

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    
    common_kwargs = dict(c_dim=25, img_resolution=512, img_channels=3)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', torch.tensor(0).to(device))

    for m in G.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    
    # Load pre-trained GenHead model
    if rank == 0:
        print(f'Resuming GenHead from "{resume_pkl_G_syn}"')
    with dnnlib.util.open_url(resume_pkl_G_syn) as f:
        G_syn = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)
        
    # G_syn = PartTriPlaneGeneratorDeform(*G_syn_meta.init_args, **G_syn_meta.init_kwargs).eval().requires_grad_(False).to(device)
    # misc.copy_params_and_buffers(G_syn_meta, G_syn, require_all=False)
    # G_syn.neural_rendering_resolution = G_syn_meta.neural_rendering_resolution
    # G_syn.rendering_kwargs = G_syn_meta.rendering_kwargs
    
    # For LPIPS loss computation
    lpips = LPIPS(net='vgg').to(device)
    set_requires_grad(lpips, requires_grad=False)
    
    # For ID loss computation
    facenet = iresnet18()
    facenet.load_state_dict(torch.load('models/arcface/ms1mv3_arcface_r18_fp16/backbone.pth'))
    facenet = facenet.eval().to(device)
    set_requires_grad(facenet, requires_grad=False)
    
    # For PD-FGC motion embedding extraction
    pd_fgc = FanEncoder()
    weight_dict = torch.load('models/pdfgc/weights/motion_model.pth')
    pd_fgc.load_state_dict(weight_dict, strict=False)
    pd_fgc = pd_fgc.eval().to(device)
    set_requires_grad(pd_fgc, requires_grad=False)
    
    # set D_patch for 3D-to-2D imitation (deprecated), see Mimic3D for details: https://github.com/SeanChenxy/Mimic3D 
    D_patch = None
    if loss_kwargs.patch_scale<1:
        img_resolution = loss_kwargs.neural_rendering_resolution_initial if loss_kwargs.neural_rendering_resolution_final is None else loss_kwargs.neural_rendering_resolution_final
        common_patch_kwargs = dict(c_dim=0, img_resolution=img_resolution, img_channels=3)
        D_patch = dnnlib.util.construct_class_by_name(**D_patch_kwargs, **common_patch_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
      
    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        load_model = [('G', G), ('G_ema', G_ema)]
        if D is not None:
            load_model.append(('D', D))
        if D_patch is not None:
            load_model.append(('D_patch', D_patch))
        for name, module in load_model:
            if name in resume_data and resume_data[name] is not None:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
            else:
                print(f'resume_data do not have {name}')
        if 'augment_pipe' in resume_data and resume_data['augment_pipe'] is not None:
            augment_p = resume_data['augment_pipe'].p

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe, lpips, D_patch]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, G_syn=G_syn, D_patch=D_patch, augment_pipe=augment_pipe, lpips=lpips, facenet=facenet, pd_fgc=pd_fgc, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    phases_asserts = [('G', G, G_opt_kwargs, G_reg_interval), ]
    if D is not None:
        phases_asserts.append(('D', D, D_opt_kwargs, D_reg_interval))
    if D_patch is not None:
        phases_asserts.append(('D_patch', D_patch, D_patch_opt_kwargs, D_patch_reg_interval))
    for name, module, opt_kwargs, reg_interval in phases_asserts:

        # if G_update_all is False:
        #     parameter_names = [n for (n, p) in module.named_parameters() if 'superresolution' not in n and not ('decoder' in n and 'encoder_global' not in n)  and 'bn' not in n] # do not update mlp and super-resolution following Real-Time Radiance Fields for Single-Image Portrait View Synthesis
        # else:
        parameter_names = [n for (n, p) in module.named_parameters() if 'bn' not in n]
        
        if name == 'G':
            parameters_group = []
            parameters_cross_names = [n for n in parameter_names if 'encoder_canonical' in n and ('maps' in n or 'maps_neutral' in n or 'proj_y' in n or 'proj_y_neutral' in n or 'norm2' in n or 'attn2' in n)]
            parameters_base_names = [n for n in parameter_names if not n in parameters_cross_names]
            parameters_cross = [p for (n, p) in module.named_parameters() if n in parameters_cross_names]
            parameters_base = [p for (n, p) in module.named_parameters() if n in parameters_base_names]
            parameters_group.append({'params': parameters_cross, 'name': 'G_cross'})
            parameters_group.append({'params': parameters_base, 'name': 'G_base'})
            parameters = parameters_group
        else:
            parameters = [p for (n, p) in module.named_parameters() if n in parameter_names]

        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(parameters, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(parameters, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
        
        if name == 'G':
            for param_group in opt.param_groups:
                if param_group['name'] == 'G_cross':
                    param_group['lr'] = param_group['lr']*cross_lr_scale
            

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')

        grid_size, shape_params, exp_params, pose_params, eye_pose_params, grid_c = setup_snapshot_image_grid(all_shape_params, all_exp_params, all_pose_params, all_eye_pose_params, all_c, static_dataset=training_set_kwargs.static)
        
        grid_z = torch.randn([shape_params.shape[0], G_syn.z_dim], device=device)
        grid_z = grid_z.unsqueeze(1).repeat(1,3,1).split(batch_gpu)

        shape_params = torch.from_numpy(shape_params).to(device).split(batch_gpu)
        exp_params = torch.from_numpy(exp_params).to(device).split(batch_gpu)
        pose_params = torch.from_numpy(pose_params).to(device).split(batch_gpu)
        eye_pose_params = torch.from_numpy(eye_pose_params).to(device).split(batch_gpu)

        grid_c = torch.from_numpy(grid_c).to(device).split(batch_gpu)

        out = []

        with torch.no_grad():
            for shape_param, exp_param, pose_param, eye_pose_param, z, c in zip(shape_params, exp_params, pose_params, eye_pose_params, grid_z, grid_c):
                shape_param = shape_param.reshape(batch_gpu*3, -1)
                exp_param = exp_param.reshape(batch_gpu*3, -1)
                pose_param = pose_param.reshape(batch_gpu*3, -1)
                eye_pose_param = eye_pose_param.reshape(batch_gpu*3, -1)
                z = z.reshape(batch_gpu*3, -1)
                z_bg = z
                c = c.reshape(batch_gpu*3, -1)

                c_cond = torch.eye(4).unsqueeze(0).repeat(batch_gpu*3,1,1).to(device)
                c_cond[...,2,3] += 4
                c_cond = torch.cat([c_cond.reshape(batch_gpu*3,-1), c[:,16:]],dim=-1)

                out.append(G_syn(shape_param, exp_param, pose_param, eye_pose_param, z=z, z_bg=z_bg, c=c, c_compose=c_cond, truncation_psi=truncation_psi, truncation_cutoff=14, noise_mode='const', run_full=True))

            images_all = torch.cat([o['image_sr'] for o in out],dim=0)
            landmarks_all = torch.cat([o['landmarks'] for o in out],dim=0)

            motions_all = loss.get_motion_feature(images_all, landmarks_all)
            motions_all = motions_all.reshape(-1, 3, motions_all.shape[-1])
            motions_app = motions_all[:,0]
            motions = motions_all[:,1]

            images_all = images_all.reshape(-1, 3, *images_all.shape[-3:])
            images_app = images_all[:,0]
            images_mot = images_all[:,1]
            images_recon = images_all[:,2]
            
            save_image_grid_all(127.5*(images_app.cpu().numpy()+1), 127.5*(images_mot.cpu().numpy()+1), 127.5*(images_recon.cpu().numpy()+1), os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)

            images_app = images_app.split(batch_gpu)
            images_mot = images_mot.split(batch_gpu)
            images_recon = images_recon.split(batch_gpu)

            motions_app = motions_app.split(batch_gpu)
            motions = motions.split(batch_gpu)

            shape_params_app = [s[:,0] for s in shape_params]
            exp_params_mot = [e[:,1] for e in exp_params]
            pose_params_mot = [p[:,1] for p in pose_params]
            eye_pose_params_mot = [p[:,1] for p in eye_pose_params]

            grid_c_recon = [c[:,2] for c in grid_c]

            images_app, images_mot, shape_params_app, exp_params_mot, pose_params_mot, eye_pose_params_mot, motions_app, motions, grid_c_recon

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    # move all params to gpu
    all_shape_params = torch.from_numpy(all_shape_params).to(device)
    all_c = torch.from_numpy(all_c).to(device)
    all_exp_params_static = torch.from_numpy(all_exp_params_static).to(device)
    all_pose_params_static = torch.from_numpy(all_pose_params_static).to(device)
    all_eye_pose_params_static = torch.from_numpy(all_eye_pose_params_static).to(device)
    all_exp_params = torch.from_numpy(all_exp_params).to(device)
    all_pose_params = torch.from_numpy(all_pose_params).to(device)
    all_eye_pose_params = torch.from_numpy(all_eye_pose_params.reshape(-1,6)).to(device)

    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_shape_params, phase_real_exp_params, phase_real_pose_params, phase_real_eye_pose_params, phase_real_c = fetch_random_params(all_shape_params, all_exp_params_static, all_pose_params_static, all_eye_pose_params_static, all_exp_params, all_pose_params, all_eye_pose_params, all_c, batchsize=batch_size_dataset, static_dataset=training_set_kwargs.static, half_static=half_static)
            
            phase_real_shape_params = phase_real_shape_params.split(batch_gpu)
            phase_real_exp_params = phase_real_exp_params.split(batch_gpu)
            phase_real_pose_params = phase_real_pose_params.split(batch_gpu)
            phase_real_eye_pose_params = phase_real_eye_pose_params.split(batch_gpu)
            phase_real_c = phase_real_c.split(batch_gpu)

            phase_gen_z = torch.randn([batch_size_dataset, G_syn.z_dim], device=device)
            phase_gen_z = phase_gen_z.unsqueeze(1).repeat(1,3,1)
            phase_gen_z = phase_gen_z.split(batch_gpu)

        #---------------------------------------------------------------------------------------------------------------------------------------
        # Online data generation. For efficiency, use same generated data for different phases
        phase_real_img_app = []
        phase_real_img_mot = []
        phase_real_img_recon = []
        phase_real_seg_recon = []
        phase_real_seg_recon_render = []
        phase_real_depth_recon = []
        phase_real_feature_recon = []
        phase_real_feature_recon_bg = []
        phase_real_triplane_recon = []
        phase_real_uv_recon = []
        phase_real_c_recon = []
        phase_real_c_compose_recon = []
        phase_real_shape_params_app = []
        phase_real_exp_params_mot = []
        phase_real_pose_params_mot = []
        phase_real_eye_pose_params_mot = []
        phase_real_motions_app = []
        phase_real_motions = []

        with torch.no_grad():
            for real_c,real_shape_params, real_exp_params, real_pose_params, real_eye_pose_params, gen_z in\
                zip(phase_real_c, phase_real_shape_params, phase_real_exp_params, phase_real_pose_params, phase_real_eye_pose_params, phase_gen_z):

                syn_out = loss.gen_data_by_G_syn(gen_z, real_shape_params, real_exp_params, real_pose_params, real_eye_pose_params, real_c)
                
                # Multiview images
                real_img = syn_out['image_sr']
                real_img = real_img.reshape(-1,3,*real_img.shape[1:])
                real_img_app = real_img[:,0]
                real_img_mot = real_img[:,1]
                real_img_recon = real_img[:,2]
                
                # Segmentation masks
                real_seg_recon = syn_out['seg']
                real_seg_recon = real_seg_recon.reshape(-1,3,*real_seg_recon.shape[1:])
                real_seg_recon = real_seg_recon[:,2]
                real_seg_recon_render = real_seg_recon
                
                # Correspondence maps, for GenHead discriminator only
                real_uv_recon = syn_out['uv']
                real_uv_recon = real_uv_recon.reshape(-1,3,*real_uv_recon.shape[1:])
                real_uv_recon = real_uv_recon[:,2]
                
                # Camera poses
                real_c_recon = syn_out['c']
                real_c_recon = real_c_recon.reshape(-1,3,*real_c_recon.shape[1:])
                real_c_recon = real_c_recon[:,2]

                real_c_compose_recon = syn_out['c_compose']
                real_c_compose_recon = real_c_compose_recon.reshape(-1,3,*real_c_compose_recon.shape[1:])
                real_c_compose_recon = real_c_compose_recon[:,2]
                
                # FLAME parameters, for head rotation only
                real_shape_params_app = real_shape_params[:,0]
                real_exp_params_mot = real_exp_params[:,1]
                real_pose_params_mot = real_pose_params[:,1]
                real_eye_pose_params_mot = real_eye_pose_params[:,1]
                
                # PD-FGC motion embeddings
                real_motions_all = syn_out['motions']
                real_motions_all = real_motions_all.reshape(-1,3,*real_motions_all.shape[1:])
                real_motions_app = real_motions_all[:,0]
                real_motions = real_motions_all[:,1]
                
                # Depth images
                real_depth_recon = syn_out['image_depth']
                real_depth_recon = real_depth_recon.reshape(-1,3,*real_depth_recon.shape[1:])
                real_depth_recon = real_depth_recon[:,2]
                
                # Feature maps before super-resolution module
                real_feature_recon = syn_out['image_feature']
                real_feature_recon = real_feature_recon.reshape(-1,3,*real_feature_recon.shape[1:])
                real_feature_recon = real_feature_recon[:,2]
                
                # Background
                real_feature_recon_bg = syn_out['background']
                real_feature_recon_bg = real_feature_recon_bg.reshape(-1,3,*real_feature_recon_bg.shape[1:])
                real_feature_recon_bg = real_feature_recon_bg[:,2]
                
                # Sampled tri-plane features
                sample_points = syn_out['coarse_sample_points']
                triplane_features = syn_out['coarse_triplane_features']
                triplane_features = torch.cat([sample_points, triplane_features], dim=-1)
                eye_masks = syn_out['eye_mask_sel']
                # print('eye_masks:',eye_masks.shape)
                mouth_masks = syn_out['mouth_mask_sel']
                # print('mouth_masks:',mouth_masks.shape)
                select_masks = ((eye_masks + mouth_masks)==0)
                
                real_triplane_recon = []
                for idx in range(len(triplane_features)):
                    select_mask = select_masks[idx].reshape(-1)
                    triplane_feature = triplane_features[idx, select_mask]
                    select_points_idx = torch.randperm(triplane_feature.shape[0])[:4000]
                    triplane_feature = triplane_feature[select_points_idx]
                    real_triplane_recon.append(triplane_feature)
                real_triplane_recon = torch.stack(real_triplane_recon, dim=0)
                real_triplane_recon = real_triplane_recon.reshape(-1,3,*real_triplane_recon.shape[1:])
                real_triplane_recon = real_triplane_recon[:,2]
                
                # print('real_triplane_recon:',real_triplane_recon.shape)
                # real_triplane_recon = None

                phase_real_img_app.append(real_img_app)
                phase_real_img_mot.append(real_img_mot)
                phase_real_img_recon.append(real_img_recon)
                phase_real_seg_recon.append(real_seg_recon)
                phase_real_seg_recon_render.append(real_seg_recon_render)
                phase_real_depth_recon.append(real_depth_recon)
                phase_real_feature_recon.append(real_feature_recon)
                phase_real_feature_recon_bg.append(real_feature_recon_bg)
                phase_real_triplane_recon.append(real_triplane_recon)
                phase_real_uv_recon.append(real_uv_recon)
                phase_real_c_recon.append(real_c_recon)
                phase_real_c_compose_recon.append(real_c_compose_recon)
                phase_real_shape_params_app.append(real_shape_params_app)
                phase_real_exp_params_mot.append(real_exp_params_mot)
                phase_real_pose_params_mot.append(real_pose_params_mot)
                phase_real_eye_pose_params_mot.append(real_eye_pose_params_mot)
                phase_real_motions_app.append(real_motions_app)
                phase_real_motions.append(real_motions)

        # Execute training phases.
        for phase in phases:
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            for real_img_app, real_img_mot, real_img_recon, real_seg_recon, real_seg_recon_render, real_depth_recon, real_feature_recon, real_feature_recon_bg, real_triplane_recon, real_uv_recon, real_c_recon, real_c_compose_recon, real_shape_params_app, real_exp_params_mot, real_pose_params_mot, real_eye_pose_params_mot, real_motions_app, real_motions in\
                zip(phase_real_img_app, phase_real_img_mot, phase_real_img_recon, phase_real_seg_recon, phase_real_seg_recon_render, phase_real_depth_recon, phase_real_feature_recon, phase_real_feature_recon_bg, phase_real_triplane_recon, phase_real_uv_recon, phase_real_c_recon, phase_real_c_compose_recon, phase_real_shape_params_app, phase_real_exp_params_mot, phase_real_pose_params_mot, phase_real_eye_pose_params_mot, phase_real_motions_app, phase_real_motions):

                loss.accumulate_gradients(phase=phase.name, real_img_app=real_img_app, real_img_mot=real_img_mot, real_img_recon=real_img_recon, real_seg_recon=real_seg_recon, real_seg_recon_render=real_seg_recon_render, real_depth_recon=real_depth_recon, real_feature_recon=real_feature_recon, real_feature_recon_bg=real_feature_recon_bg, real_triplane_recon=real_triplane_recon, real_uv_recon=real_uv_recon, real_c_recon=real_c_recon,\
                    real_c_compose_recon=real_c_compose_recon, shape_params_app=real_shape_params_app, exp_params_mot=real_exp_params_mot, pose_params_mot=real_pose_params_mot, eye_pose_params_mot=real_eye_pose_params_mot, motions_app=real_motions_app, motions=real_motions, gain=phase.interval, cur_nimg=cur_nimg, motion_scale=motion_scale, swapping_prob=swapping_prob, half_static=half_static)

            phase.module.requires_grad_(False)
                

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                # Do not update mlp decoder and super-resolution module at the warm-up stage following Live3dportrait: https://arxiv.org/abs/2305.02310 
                if cur_nimg <= loss.discrimination_kimg * 1e3 and phase.name == 'G':
                    sub_params = [p for (n, p) in phase.module.named_parameters() if 'superresolution' in n or ('decoder' in n and 'encoder_global' not in n)] 
                    for param in sub_params:
                        if param.grad is not None:
                            param.grad.zero_()
                
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            print('Saving images...')
            out = []
            for image_app, image_mot, shape_param_app, exp_param_mot, pose_param_mot, eye_pose_param_mot, motion_app, motion, c in zip(images_app, images_mot, shape_params_app, exp_params_mot, pose_params_mot, eye_pose_params_mot, motions_app, motions, grid_c_recon):
                with torch.no_grad():
                    out.append(G_ema(image_app, image_mot, motion_app, motion, shape_param_app, exp_param_mot, pose_param_mot, eye_pose_param_mot, c=c, patch_scale=loss_kwargs['patch_scale'], neural_rendering_resolution_patch=loss_kwargs.neural_rendering_resolution_final, run_full=True, motion_scale=motion_scale))
            if 'image' in out[0]:
                images = torch.cat([o['image'].cpu() for o in out]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            if 'uv' in out[0]:
                uv_images = torch.cat([o['uv'].cpu() for o in out]).numpy()
                save_image_grid(uv_images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_uv.png'), drange=[-1,1], grid_size=grid_size)
            if 'seg' in out[0]:
                seg_images = torch.cat([o['seg'].cpu() for o in out]).numpy()
                save_image_grid(seg_images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_seg.png'), drange=[-1,1], grid_size=grid_size)
            if 'image_depth' in out[0]:
                images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
                save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
            if 'image_sr' in out[0] and out[0]['image_sr'] is not None:
                images_sr = torch.cat([o['image_sr'].cpu() for o in out]).numpy()
                save_image_grid(images_sr, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_sr.png'), drange=[-1,1], grid_size=grid_size)
                

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('D_patch', D_patch), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
