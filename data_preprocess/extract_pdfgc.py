import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../portrait4d/')))

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from portrait4d.training.utils.preprocess import estimate_norm_torch_pdfgc
from portrait4d.models.pdfgc.encoder import FanEncoder
from kornia.geometry import warp_affine
import torch.nn.functional as F

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

@click.command()
@click.option('--input_dir', help='Where to save the output images', type=str, default='', metavar='DIR')
@click.option('--save_dir', help='Where to save the motion embeddings', type=str, default='', metavar='DIR')
def extract_motion_feature(
    input_dir: str,
    save_dir: str,
):
    device = torch.device('cuda')
    
    save_dir_mot = os.path.join(save_dir,'motion_feats')
    os.makedirs(save_dir_mot, exist_ok=True)
    
    # load motion encoder
    pd_fgc = FanEncoder()
    weight_dict = torch.load('../portrait4d/models/pdfgc/weights/motion_model.pth')
    pd_fgc.load_state_dict(weight_dict, strict=False)
    pd_fgc = pd_fgc.eval().to(device)    

    img_list = sorted(os.listdir(os.path.join(input_dir,'align_images')))

    with torch.no_grad():
        for idx, img_name in enumerate(img_list):
            print('Extracting motion embedding for image %s (%d/%d) ...' % (img_name, idx, len(img_list)))
            
            # source image
            img = np.array(PIL.Image.open(os.path.join(input_dir, 'align_images', img_name)))
            img = torch.from_numpy((img.astype(np.float32)/127.5 - 1)).to(device)
            img = img.permute([2,0,1]).unsqueeze(0)
            
            # source landmarks: y axis points downwards
            lmks = np.load(os.path.join(input_dir,'3dldmks_align',img_name.replace('.png','.npy').replace('.jpg','.npy')))
            lmks = torch.from_numpy(lmks).to(device).unsqueeze(0)
            
            # calculate motion embedding
            motion = get_motion_feature(pd_fgc, img, lmks).squeeze(0).cpu().numpy()
            np.save(os.path.join(save_dir_mot,img_name.replace('.png','.npy').replace('.jpg','.npy')), motion)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    extract_motion_feature() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
