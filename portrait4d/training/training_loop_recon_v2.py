# Main training loop of Portrait4D-v2, modified from EG3D: https://github.com/NVlabs/eg3d

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
from camera_utils import LookAtPoseSampler
from models.lpips.lpips import LPIPS
from models.arcface.iresnet import iresnet18
from models.pdfgc.encoder import FanEncoder

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(1920 // training_set.image_shape[2], 7, 8)
    gh = np.clip(1080 // training_set.image_shape[1], 4, 8)

    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images_app, images_mot, images_recon, segs_recon, segs_recon_render, depths_recon, features_recon, features_recon_bg, triplanes_recon, labels_recon, shape_params_app, exp_params_mot, pose_params_mot, eye_pose_params_mot, motions_app, motions = zip(*[training_set[i] for i in grid_indices])

    return (gw, gh), np.stack(images_app), np.stack(images_mot), np.stack(images_recon), np.stack(segs_recon), np.stack(shape_params_app), np.stack(exp_params_mot), np.stack(pose_params_mot), np.stack(eye_pose_params_mot), np.stack(labels_recon), np.stack(motions_app), np.stack(motions)

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
    resume_pkl_G_fix        = None,     # Checkpoint of pre-trained static reconstructor for multi-view video synthesis.
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

    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size_dataset, **data_loader_kwargs))

    # validation_set = dnnlib.util.construct_class_by_name(**validation_set_kwargs) # subclass of training.dataset.Dataset

    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    
    common_kwargs = dict(c_dim=25, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))

    for m in G.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Use static reconstructor learned on synthetic data for multi-view regularization
    if resume_pkl_G_fix is not None:
        G_fix = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).eval().requires_grad_(False).to(device)
        G_fix.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    else:
        G_fix = None
    
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


    D_patch = None
    if loss_kwargs.patch_scale<1:
        img_resolution = loss_kwargs.neural_rendering_resolution_initial if loss_kwargs.neural_rendering_resolution_final is None else loss_kwargs.neural_rendering_resolution_final
        common_patch_kwargs = dict(c_dim=0, img_resolution=img_resolution, img_channels=training_set.num_channels)
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
    
    # Load pre-trained static reconstructor
    if (resume_pkl_G_fix is not None) and (rank == 0):
        print(f'Resuming G_fix from "{resume_pkl_G_fix}"')
        with dnnlib.util.open_url(resume_pkl_G_fix) as f:
            resume_data = legacy.load_network_pkl(f)
        load_model = [('G_ema', G_fix)]
        for name, module in load_model:
            if name in resume_data and resume_data[name] is not None:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
            else:
                print(f'resume_data do not have {name}')    

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
    for module in [G, D, G_ema, augment_pipe, lpips, D_patch, G_fix]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, D_patch=D_patch, augment_pipe=augment_pipe, lpips=lpips, facenet=facenet, G_fix=G_fix, pd_fgc=pd_fgc, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    phases_asserts = [('G', G, G_opt_kwargs, G_reg_interval), ]
    if D is not None:
        phases_asserts.append(('D', D, D_opt_kwargs, D_reg_interval))
    if D_patch is not None:
        phases_asserts.append(('D_patch', D_patch, D_patch_opt_kwargs, D_patch_reg_interval))
    for name, module, opt_kwargs, reg_interval in phases_asserts:

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
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images_app, images_mot, images_recon, segs_recon, shape_params_app, exp_params_mot, pose_params_mot, eye_pose_params_mot, labels_recon, motions_app, motions = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid_all(images_app, images_mot, images_recon, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)

        images_app = (torch.from_numpy(images_app).to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
        images_mot = (torch.from_numpy(images_mot).to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)

        grid_c_compose_recon = torch.from_numpy(labels_recon).to(device).split(batch_gpu)

        shape_params_app = torch.from_numpy(shape_params_app).to(device).split(batch_gpu)
        exp_params_mot = torch.from_numpy(exp_params_mot).to(device).split(batch_gpu)
        pose_params_mot = torch.from_numpy(pose_params_mot).to(device).split(batch_gpu)
        eye_pose_params_mot = torch.from_numpy(eye_pose_params_mot).to(device).split(batch_gpu)
        motions_app = torch.from_numpy(motions_app).to(device).split(batch_gpu)
        motions = torch.from_numpy(motions).to(device).split(batch_gpu)

        grid_c_recon = []
        for _grid_c_compose_recon,_shape_params_app,_exp_params_mot,_pose_params_mot in zip(grid_c_compose_recon,shape_params_app,exp_params_mot,pose_params_mot):
            _grid_c_recon = G.deformer.flame_deform.decompose_camera_pose(_grid_c_compose_recon,_shape_params_app,_exp_params_mot,_pose_params_mot)
            grid_c_recon.append(_grid_c_recon)
        grid_c_recon = tuple(grid_c_recon)

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
    
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):

            phase_real_img_app, phase_real_img_mot, phase_real_img_recon, phase_real_seg_recon, phase_real_seg_recon_render, phase_real_depth_recon, phase_real_feature_recon, phase_real_feature_recon_bg, phase_real_triplane_recon, phase_real_c_compose_recon, phase_real_shape_params_app, phase_real_exp_params_mot, phase_real_pose_params_mot, phase_real_eye_pose_params_mot, phase_real_motions_app, phase_real_motions = next(training_set_iterator)
            
            phase_real_img_app = (phase_real_img_app.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_img_mot = (phase_real_img_mot.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_img_recon = (phase_real_img_recon.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_seg_recon = (phase_real_seg_recon.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_seg_recon_render = (phase_real_seg_recon_render.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_depth_recon = phase_real_depth_recon.to(device).to(torch.float32).split(batch_gpu)
            phase_real_feature_recon = phase_real_feature_recon.to(device).to(torch.float32).split(batch_gpu)
            phase_real_feature_recon_bg = phase_real_feature_recon_bg.to(device).to(torch.float32).split(batch_gpu)
            phase_real_triplane_recon = phase_real_triplane_recon.to(device).to(torch.float32).split(batch_gpu)
            phase_real_c_compose_recon = phase_real_c_compose_recon.to(device)
            phase_real_motions_app = phase_real_motions_app.to(device)
            phase_real_motions = phase_real_motions.to(device)

            phase_real_shape_params_app = phase_real_shape_params_app.to(device).to(torch.float32)
            phase_real_exp_params_mot = phase_real_exp_params_mot.to(device).to(torch.float32)
            phase_real_pose_params_mot = torch.cat([phase_real_pose_params_mot[...,:3],phase_real_pose_params_mot[...,3:]],dim=-1).to(device).to(torch.float32)
            phase_real_eye_pose_params_mot = phase_real_eye_pose_params_mot.to(device).to(torch.float32)

            phase_real_c_recon = G.deformer.flame_deform.decompose_camera_pose(phase_real_c_compose_recon,phase_real_shape_params_app,phase_real_exp_params_mot,phase_real_pose_params_mot)
            phase_real_uv_recon = G.deformer.renderer(phase_real_shape_params_app, phase_real_exp_params_mot, phase_real_pose_params_mot, phase_real_eye_pose_params_mot, phase_real_c_recon, half_size=int(G.img_resolution/2))[0]

            phase_real_shape_params_app = phase_real_shape_params_app.split(batch_gpu)
            phase_real_exp_params_mot = phase_real_exp_params_mot.split(batch_gpu)
            phase_real_pose_params_mot = phase_real_pose_params_mot.split(batch_gpu)
            phase_real_eye_pose_params_mot = phase_real_eye_pose_params_mot.split(batch_gpu)
            phase_real_c_compose_recon = phase_real_c_compose_recon.split(batch_gpu)
            phase_real_c_recon = phase_real_c_recon.split(batch_gpu)
            phase_real_uv_recon = phase_real_uv_recon.split(batch_gpu)
            phase_real_motions_app = phase_real_motions_app.split(batch_gpu)
            phase_real_motions = phase_real_motions.split(batch_gpu)
            #---------------------------------------------------------------------------------------------------------------------------------------

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

                if len(real_seg_recon_render.shape) == 1:
                    real_seg_recon_render = None                
                if len(real_depth_recon.shape) == 1:
                    real_depth_recon = None
                if len(real_feature_recon.shape) == 1:
                    real_feature_recon = None
                if len(real_feature_recon_bg.shape) == 1:
                    real_feature_recon_bg = None 
                if len(real_triplane_recon.shape) == 1:
                    real_triplane_recon = None
                loss.accumulate_gradients(phase=phase.name, real_img_app=real_img_app, real_img_mot=real_img_mot, real_img_recon=real_img_recon, real_seg_recon=real_seg_recon, real_seg_recon_render=real_seg_recon_render, real_depth_recon=real_depth_recon, real_feature_recon=real_feature_recon, real_feature_recon_bg=real_feature_recon_bg, real_triplane_recon=real_triplane_recon, real_uv_recon=real_uv_recon, real_c_recon=real_c_recon,\
                    real_c_compose_recon=real_c_compose_recon, shape_params_app=real_shape_params_app, exp_params_mot=real_exp_params_mot, pose_params_mot=real_pose_params_mot, eye_pose_params_mot=real_eye_pose_params_mot, motions_app=real_motions_app, motions=real_motions, gain=phase.interval, cur_nimg=cur_nimg, motion_scale=motion_scale, swapping_prob=swapping_prob, half_static=half_static)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
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
                
            #--------------------

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
