# Inference script for Portrait4D and Portrait4D-v2

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
import json

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.reconstructor.triplane_reconstruct import TriPlaneReconstructorNeutralize
from training.utils.preprocess import estimate_norm_torch_pdfgc
from models.pdfgc.encoder import FanEncoder
from kornia.geometry import warp_affine
import torch.nn.functional as F
import pickle
from pytorch3d.io import load_obj, save_obj
from shape_utils import convert_sdf_samples_to_ply

#----------------------------------------------------------------------------

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

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')
#----------------------------------------------------------------------------

def get_motion_feature(pd_fgc, imgs, lmks, crop_size=224, crop_len=16, reverse_y=False):

    trans_m = estimate_norm_torch_pdfgc(lmks, imgs.shape[-1], reverse_y=reverse_y)
    imgs_warp = warp_affine(imgs, trans_m, dsize=(224, 224))
    imgs_warp = imgs_warp[:,:,:crop_size - crop_len*2, crop_len:crop_size - crop_len]
    imgs_warp = torch.clamp(F.interpolate(imgs_warp,size=[crop_size,crop_size],mode='bilinear'),-1,1)
    
    out = pd_fgc(imgs_warp)
    motions = torch.cat([out[1],out[2],out[3]],dim=-1)

    return motions
#----------------------------------------------------------------------------

def pose2rot(pose):
    rot_mats = batch_rodrigues(
        pose.view(-1, 3), dtype=pose.dtype).view([pose.shape[0], 3, 3])
    
    return rot_mats

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='./pretrained_models/portrait4d-v2-vfhq512.pkl')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--srcdir', help='Where to save the output images', type=str, default='./examples', metavar='DIR')
@click.option('--tardir', help='Where to save the output images', type=str, default='./examples', metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, default='experiments/portrait4d-reenacted', metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=True, metavar='BOOL', default=True, show_default=True)
@click.option('--render_size', help='', type=int, required=False, metavar='int', default=64, show_default=True)
@click.option('--chunk', help='', type=int, required=False, metavar='int', default=None, show_default=True)
@click.option('--use_neck', help='Use neck rotation?', type=bool, required=True, metavar='BOOL', default=True, show_default=True)
@click.option('--use_simplified', help='Use simplified flame parameters for head pose control?', type=bool, required=True, metavar='BOOL', default=False, show_default=True)
@click.option('--shape', help='Extract shapes using marching cubes?', type=bool, required=True, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    srcdir: str,
    tardir: str,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    render_size: int,
    chunk: int,
    use_neck: bool,
    use_simplified: bool,
    shape: bool
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
        G_new = TriPlaneReconstructorNeutralize(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=False)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new  
    
    # load motion encoder
    pd_fgc = FanEncoder()
    weight_dict = torch.load('models/pdfgc/weights/motion_model.pth')
    pd_fgc.load_state_dict(weight_dict, strict=False)
    pd_fgc = pd_fgc.eval().to(device)    

    os.makedirs(outdir, exist_ok=True)
    
    src_img_list = sorted(os.listdir(os.path.join(srcdir,'align_images')))
    tar_img_list = sorted(os.listdir(os.path.join(tardir,'align_images')))

    with torch.no_grad():
        for idx, src_img_name in enumerate(src_img_list):
            print('Generating results for image %s (%d/%d) ...' % (src_img_name, idx, len(src_img_list)))
            torch.manual_seed(idx)

            outdir_sub = os.path.join(outdir,f"{src_img_name.replace('.png','').replace('.jpg','')}")
            os.makedirs(outdir_sub, exist_ok=True)
            
            # source image
            img_app = np.array(PIL.Image.open(os.path.join(srcdir, 'align_images', src_img_name)))
            img_app = torch.from_numpy((img_app.astype(np.float32)/127.5 - 1)).to(device)
            img_app = img_app.permute([2,0,1]).unsqueeze(0)
            
            # source landmarks: y axis points downwards
            lmks_app = np.load(os.path.join(srcdir,'3dldmks_align',src_img_name.replace('.png','.npy').replace('.jpg','.npy')))
            lmks_app = torch.from_numpy(lmks_app).to(device).unsqueeze(0)
            
            # calculate motion embedding
            motion_app = get_motion_feature(pd_fgc, img_app, lmks_app)
            
            # source flame params
            if use_simplified:
                params_app = np.load(os.path.join(srcdir,'bfm2flame_params_simplified',src_img_name.replace('.png','.npy').replace('.jpg','.npy')))
            else:
                params_app = np.load(os.path.join(srcdir,'flame_optim_params',src_img_name.replace('.png','.npy').replace('.jpg','.npy')))
            params_app = torch.from_numpy(params_app).to(device).reshape(1,-1)
            
            shape_params_app = params_app[:,25:325]
            exp_params_app = params_app[:,325:425]
            pose_params_app = params_app[:,425:431]
            eye_pose_app = params_app[:,431:437]
            
            for tar_idx, tar_img_name in enumerate(tar_img_list):
                
                # target image
                img_mot = np.array(PIL.Image.open(os.path.join(tardir, 'align_images', tar_img_name)))
                img_mot = torch.from_numpy((img_mot.astype(np.float32)/127.5 - 1)).to(device)
                img_mot = img_mot.permute([2,0,1]).unsqueeze(0)
                
                # target landmarks: y axis points downwards
                lmks_mot = np.load(os.path.join(tardir,'3dldmks_align',tar_img_name.replace('.png','.npy').replace('.jpg','.npy')))
                lmks_mot = torch.from_numpy(lmks_mot).to(device).unsqueeze(0)
                
                # calculate motion embedding
                motion_mot = get_motion_feature(pd_fgc, img_mot, lmks_mot)
                
                # target flame params
                if use_simplified:
                    params_mot = np.load(os.path.join(tardir,'bfm2flame_params_simplified',tar_img_name.replace('.png','.npy').replace('.jpg','.npy')))
                else:
                    params_mot = np.load(os.path.join(tardir,'flame_optim_params',tar_img_name.replace('.png','.npy').replace('.jpg','.npy')))
                params_mot = torch.from_numpy(params_mot).to(device).reshape(1,-1)
                
                shape_params_mot = params_mot[:,25:325]
                exp_params_mot = params_mot[:,325:425]
                pose_params_mot = params_mot[:,425:431]
                eye_pose_mot = params_mot[:,431:437]            
            
                c = params_mot[:,:25]
                intrinsics = c[:,16:]
                        
                if use_neck:
                    extrinsics = torch.eye(4).to(device)
                    extrinsics[1,3] = 0.01
                    extrinsics[2,3] = 4.2
                    c = torch.cat([extrinsics.reshape(1,-1), intrinsics.reshape(1,-1)], dim=-1)
                else:
                    pose_params_mot[:,:3] *= 0
                
                _deformer = G._deformer(shape_params_app,exp_params_mot,pose_params_mot,eye_pose_mot,use_rotation_limits=False, smooth_th=3e-3)
                out = G.synthesis(img_app, img_mot, motion_app, motion_mot, c, _deformer=_deformer, neural_rendering_resolution=128, motion_scale=1)
                
                
                img = out['image_sr']

                img_ = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_mot_ = (img_mot.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_app_ = (img_app.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                img = torch.cat([img_app_,img_,img_mot_],dim=2)

                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir_sub}/{tar_idx:05d}.jpg', quality=95)
                
                if shape:
                    os.makedirs(f'{outdir_sub}/shapes', exist_ok=True)
                    
                    max_batch=1000000
                    shape_res = 128 # Marching cube resolution

                    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
                    samples = samples.to(c.device)
                    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=c.device)
                
                    head = 0
                    with tqdm(total = samples.shape[1]) as pbar:
                        with torch.no_grad():
                            while head < samples.shape[1]:
                                torch.manual_seed(0)
                                out = G.sample_mixed(img_app, img_mot, motion_app, motion_mot, samples[:, head:head+max_batch], torch.zeros_like(samples[:, head:head+max_batch]), shape_params_app, exp_params_mot, pose_params_mot, eye_pose_mot)
                                sigma = out['sigma']
                                sigmas[:, head:head+max_batch] = sigma
                                head += max_batch
                                pbar.update(max_batch)

                    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
                    sigmas = np.flip(sigmas, 0)

                    convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), voxel_origin, voxel_size, f'{outdir_sub}/shapes/{tar_idx:05d}.ply', level=25)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
