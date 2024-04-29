# Inference script for GenHead

"""Generate images and shapes using pretrained network pickle."""
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import cv2
import imageio

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.generator.triplane import PartTriPlaneGeneratorDeform
import torch.nn.functional as F
import pickle

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='./pretrained_models/genhead-ffhq512.pkl')
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', default='0-10')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, default='./experiments/genhead-syn', metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=True, metavar='BOOL', default=True, show_default=True)
@click.option('--render_size', help='', type=int, required=False, metavar='int', default=256, show_default=True)
@click.option('--chunk', help='', type=int, required=False, metavar='int', default=None, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    render_size: int,
    chunk: int
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        Network = legacy.load_network_pkl(f) # type: ignore
        G = Network['G_ema'].to(device)

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = PartTriPlaneGeneratorDeform(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=False)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    
    # Load valid FLAME parameters set
    shape_n_c_params = np.load('./data/ffhq_all_shape_n_c_params.npy').reshape(-1,325)
    motion_params = np.load('./data/ffhq_all_motion_params.npy').reshape(-1,112)
    
    # Pre-define neck rotation
    mv_len = 40
    neck_pitch = [0.2 * np.cos(t * 2 * np.pi)+0.1  for t in np.linspace(0, 1, mv_len)]
    neck_yaw = [0.4 * np.sin(t * 2 * np.pi)  for t in np.linspace(0, 1, mv_len)]
    
    os.makedirs(outdir, exist_ok=True)

    # Generate images.
    with torch.no_grad():
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))

            torch.manual_seed(seed)
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            
            c = torch.from_numpy(shape_n_c_params[seed, :25].astype(np.float32)).unsqueeze(0).to(device)
            shape_param = torch.from_numpy(shape_n_c_params[seed, 25:325].astype(np.float32)).unsqueeze(0).to(device)
            exp_param = torch.from_numpy(motion_params[seed, :100].astype(np.float32)).unsqueeze(0).to(device)
            pose_param = torch.from_numpy(motion_params[seed, 100:106].astype(np.float32)).unsqueeze(0).to(device)
            eyepose_param = torch.from_numpy(motion_params[seed, 106:112].astype(np.float32)).unsqueeze(0).to(device)

            c_cond_extrinsics = torch.eye(4).unsqueeze(0).to(device)
            c_cond_extrinsics[...,2,3] += 4
            c_cond = torch.cat([c_cond_extrinsics.reshape(1,-1), c[:, 16:].reshape(1,-1)], dim=-1)
            
            c_render = c_cond

            if G.flame_condition:
                c_cond = torch.cat([c_cond,shape_param],dim=-1)

            for i in range(mv_len):
                pose_param[:,0] = neck_pitch[i]
                pose_param[:,1] = neck_yaw[i]
                _deformer = G._deformer(shape_param,exp_param,pose_param,eyepose_param,use_rotation_limits=False)

                ws = G.mapping(z, c_cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                out = G.synthesis(ws, z, c_render, _deformer, neural_rendering_resolution=64, noise_mode='const', smpl_param=(shape_param,exp_param,pose_param,eyepose_param))
                img = out['image_sr']

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{seed:04d}_{i:04d}.png')
            

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
