# Training script of Portrait4D, modified from EG3D: https://github.com/NVlabs/eg3d

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train a 4D head reconstructor using the techniques described in the paper
"Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data."
"""
from configs import cfg as opts

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop_recon_v1 as training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, class_name='training.dataloader.dataset_recon.GenHeadReconSegLmdbFolderDatasetV2New'):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name=class_name, resolution=512, data_type='vfhq', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

def main(**kwargs):

    # Initialize config.
    # opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    if opts.use_flame_mot:
        mot_dims = 109
    else:
        mot_dims = 548
    c.G_kwargs = dnnlib.EasyDict(class_name=None, mot_dims=mot_dims, deformation_kwargs=dnnlib.EasyDict(flame_cfg_path='models/FLAME/cfg.yaml'))
    c.D_kwargs = dnnlib.EasyDict(class_name=None, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.D_patch_kwargs = dnnlib.EasyDict(class_name=None, block_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.9,0.999], eps=1e-5)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.9,0.999], eps=1e-5)
    c.D_patch_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-5)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.loss_recon_v1.AnimatablePortraitReconLoss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    # c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    c.training_set_kwargs = dnnlib.EasyDict(class_name='online', resolution=512, data_type=None, path='', use_labels=True, max_size=None, xflip=False)
    dataset_name = 'online'
    if opts.static:
        dataset_name += '-static'

    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    c.training_set_kwargs.static = opts.static # set true for static multiview reconstruction
    c.training_set_kwargs.use_flame_mot = opts.use_flame_mot
    c.training_set_kwargs.shape_n_c_params_path1 = opts.shape_n_c_params_path1
    c.training_set_kwargs.shape_n_c_params_path2 = opts.shape_n_c_params_path2
    c.training_set_kwargs.motion_params_path1 = opts.motion_params_path1
    c.training_set_kwargs.motion_params_path2 = opts.motion_params_path2

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.resume_kimg = opts.resume_kimg
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax

    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.D_patch_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_patch_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.loss_kwargs.r1_gamma_uv = opts.gamma_uv
    c.loss_kwargs.r1_gamma_seg = opts.gamma_seg
    c.loss_kwargs.r1_gamma_patch = opts.gamma_patch
    c.loss_kwargs.truncation_psi = opts.truncation_psi
    c.G_opt_kwargs.lr = opts.glr
    c.cross_lr_scale = opts.cross_lr_scale

    c.motion_scale = 1 if opts.static is False else 0
    c.swapping_prob = None
    c.half_static = True if opts.static is False else False
    c.truncation_psi = opts.truncation_psi

    c.D_opt_kwargs.lr = opts.dlr
    c.D_patch_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = opts.snap
    c.network_snapshot_ticks = 100
    # c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = opts.g_module
    c.D_kwargs.class_name = opts.d_module
    if opts.patch_scale<1:
        c.D_patch_kwargs.class_name = opts.d_patch_module
    c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.filter_mode = 'antialiased' # Filter mode for raw images ['antialiased', 'none', float [0-1]]
    c.D_kwargs.disc_c_noise = opts.disc_c_noise # Regularization for discriminator pose conditioning

    c.G_kwargs.flame_full = opts.g_flame_full
    c.G_kwargs.has_superresolution = opts.g_has_superresolution
    c.G_kwargs.has_background = opts.g_has_background
    c.G_kwargs.num_blocks_neutral = opts.g_num_blocks_neutral
    c.G_kwargs.num_blocks_motion = opts.g_num_blocks_motion
    c.G_kwargs.motion_map_layers = opts.g_motion_map_layers

    c.D_kwargs.has_superresolution = opts.d_has_superresolution
    c.D_kwargs.has_uv = opts.d_has_uv
    c.D_kwargs.has_seg = opts.d_has_seg
    c.D_patch_kwargs.has_superresolution = False
    c.D_patch_kwargs.has_uv = False
    c.D_patch_kwargs.has_seg = False

    if c.training_set_kwargs.resolution == 512:
        sr_module = 'models.stylegan.superresolution.SuperresolutionHybrid8XDC'
    elif c.training_set_kwargs.resolution == 256:
        sr_module = 'models.stylegan.superresolution.SuperresolutionHybrid4X'
    elif c.training_set_kwargs.resolution == 128:
        sr_module = 'models.stylegan.superresolution.SuperresolutionHybrid2X'
    elif c.training_set_kwargs.resolution == 64:
        sr_module = None
    else:
        assert False, f"Unsupported resolution {c.training_set_kwargs.resolution}; make a new superresolution module"
    
    if opts.sr_module != None:
        sr_module = opts.sr_module
    
    rendering_options = {
        'image_resolution': c.training_set_kwargs.resolution,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': sr_module,
        'c_gen_conditioning_zero': not opts.gen_pose_cond, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale': opts.c_scale, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': opts.sr_noise_mode, # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': opts.density_reg, # strength of density regularization
        'density_reg_p_dist': opts.density_reg_p_dist, # distance at which to sample perturbed points for density regularization
        'reg_type': opts.reg_type, # for experimenting with variations on density regularization
        'decoder_lr_mul': opts.decoder_lr_mul, # learning rate multiplier for decoder
        'sr_antialias': True,
    }

    rendering_options.update({
        'depth_resolution': 48, # number of uniform samples to take per ray.
        'depth_resolution_importance': 48, # number of importance samples to take per ray.
        'ray_start': 'auto-plane', # near point along each ray to start taking samples.
        'ray_end': 'auto-plane', # far point along each ray to stop taking samples. 
        'box_warp': 1.3, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
        'avg_camera_radius': 1.0, # used only in the visualizer to specify camera orbit radius.
        'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
    })


    if opts.density_reg > 0:
        c.G_reg_interval = opts.density_reg_every
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.deformation_kwargs.box_warp = rendering_options['box_warp']
    c.G_kwargs.num_fp16_res = 0
    c.loss_kwargs.blur_init_sigma = 0 # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_patch_seg = opts.blur_patch_seg
    c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32 # Fade out the blur during the first N kimg.
    c.loss_kwargs.discrimination_kimg = 1000 # where to add adversarial loss
    c.loss_kwargs.gmain = 0.01

    c.loss_kwargs.gpc_reg_prob = opts.gpc_reg_prob if opts.gen_pose_cond else None
    c.loss_kwargs.gpc_reg_fade_kimg = opts.gpc_reg_fade_kimg
    c.loss_kwargs.dual_discrimination = True
    c.loss_kwargs.neural_rendering_resolution_initial = opts.neural_rendering_resolution_initial
    c.loss_kwargs.neural_rendering_resolution_final = opts.neural_rendering_resolution_final
    c.loss_kwargs.neural_rendering_resolution_fade_kimg = opts.neural_rendering_resolution_fade_kimg
    c.loss_kwargs.patch_scale = opts.patch_scale
    c.loss_kwargs.patch_gan = opts.patch_gan
    c.G_kwargs.sr_num_fp16_res = opts.sr_num_fp16_res
    c.G_kwargs.add_block = opts.patch_scale<1
    c.G_kwargs.masked_sampling = opts.masked_sampling
    c.loss_kwargs.masked_sampling = opts.masked_sampling
    c.loss_kwargs.perturb_params = True
    

    c.G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=opts.cbase, channel_max=opts.cmax, fused_modconv_default='inference_only', use_ws_ones=opts.use_ws_ones)

    c.loss_kwargs.style_mixing_prob = opts.style_mixing_prob

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        if not opts.resume_blur:
            c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
            c.loss_kwargs.gpc_reg_fade_kimg = 0 # Disable swapping rampup
    
    if opts.resume_syn is not None:
        c.resume_pkl_G_syn = opts.resume_syn

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None

    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
