import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../portrait4d/')))
from portrait4d.training.dataloader.protocols import datum_genhead_pb2 as datum_pb2

import cv2
import numpy as np
from tqdm import tqdm
import lmdb
import json
import torch
import subprocess
import click


def cam_world_matrix_transform(RT):

    #RT (B,4,4) cam2world matrix or world2cam matrix

    rot = RT[:,:3,:3]
    trans = RT[:,:3,3:]

    inverse_RT = torch.eye(4,device=RT.device).unsqueeze(0).repeat(RT.shape[0], 1, 1)
    inverse_rot = rot.permute(0,2,1)
    inverse_trans = - inverse_rot @ trans
    inverse_RT[:,:3,:3] = inverse_rot
    inverse_RT[:,:3,3:] = inverse_trans

    return inverse_RT

# def valid_label(label):
#     extrinsics = torch.from_numpy(label[:16]).reshape(-1,4,4)
#     inverse_extrinsics = cam_world_matrix_transform(extrinsics)
#     inverse_trans = inverse_extrinsics[:,:3,3]*3
#     inverse_trans_z = torch.abs(inverse_trans[:,2]).unsqueeze(0)
#     if inverse_trans_z > 10:
#         return False
#     else:
#         return True

def array_to_datum(image, seg, label, mot):

    datum = datum_pb2.Datum_genhead()
    datum.width, datum.height, datum.channels = 512, 512, 3

    datum.image = image
    datum.seg = seg
    datum.labels = bytes(label)
    datum.mots = bytes(mot)

    return datum


def _save_to_lmdb(save_path, data_list, rescale_camera=False):
    db = lmdb.open(save_path, map_size=1024 ** 4 * 5)
    txn = db.begin(write=True)
    count = 0
    for data_item in tqdm(data_list):

        img_path = data_item[0]
        seg_path = data_item[1]
        label_path = data_item[2]
        mot_path = data_item[3]
        
        if not os.path.isfile(mot_path):
            print('pass')
            continue

        label = np.load(label_path)
        
        if rescale_camera:
            intrinsics = label[16:25].reshape(3,3)

            if intrinsics[0,2]*2 != 512:
                intrinsics[:2,:] *= (0.5*512/intrinsics[0,2])

            intrinsics[0, 0] /= 512
            intrinsics[1, 1] /= 512
            intrinsics[0, 2] /= 512
            intrinsics[1, 2] /= 512   
            
            # rescale extrinsics
            extrinsics = label[:16].reshape(4,4)
            extrinsics[:3,3] *= 3
        
        mot = np.load(mot_path)
        mot_select = mot
        # mot_eye = mot[2048+25088+6:2048+25088+6+6]
        # mot_emo = mot[2048+25088+6+6:2048+25088+6+6+30]
        # mot_mouth = mot[2048+25088+6+6+30:2048+25088+6+6+30+512]
        # mot_select = np.concatenate([mot_eye,mot_emo,mot_mouth],axis=0)
        
        with open(img_path, 'rb') as f:
            image = f.read()
        with open(seg_path, 'rb') as f:
            seg = f.read()
        datum = array_to_datum(
            image, seg, label, mot_select)
        txn.put('{:0>8d}'.format(count).encode(), datum.SerializeToString())
        count += 1

        if count%200 == 0:
            txn.commit()
            txn = db.begin(write=True)

    print('num_samples: ', count)
    txn.put('num_samples'.encode(), str(count).encode())
    txn.commit()
    db.close()
    
    return count


@click.command()
@click.option('--data_dir', type=str, help='Input folder', default='')
@click.option('--output_dir', type=str, help='Output folder', default='../../portrait4d/data/')
@click.option('--resolution', type=int, help='Image resolution', default=512)
@click.option('--max_len', type=int, help='Number of subjects', default=-1)

def main(data_dir:str, output_dir:str, resolution:int, max_len:int):
    
    save_path = os.path.join(output_dir, "FFHQ")
    
    # Rebalance the pose distribution of FFHQ
    f = open('ffhq_rebalance.json','r')
    dataset_replicate = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_items_list = []
    
    # 00000-69000 contain original FFHQ images and 70000-139000 contain their horizontal-flipped ones.
    idxs = list(range(0,140)) 

    for idx in tqdm(idxs):
        subdir = '%02d000'%idx
        for img_item in sorted(os.listdir(os.path.join(data_dir, subdir,'align_images'))):
            img_name = img_item.split('.')[0]
            img_path = os.path.join(data_dir, subdir, 'align_images', img_item)
            seg_path = os.path.join(data_dir, subdir, 'segs', img_item)
            label_path = os.path.join(data_dir, subdir, 'flame_optim_params', img_name+'.npy')
            mot_path = os.path.join(data_dir, subdir, 'motion_feats', img_name+'.npy')
            
            replicate_num = dataset_replicate[str(int(img_name)%70000)]
            for i in range(replicate_num):
                data_items_list.append([img_path, seg_path, label_path, mot_path])
    
    if not max_len == -1:
        data_items_list = data_items_list[:max_len]
    data_count = len(data_items_list)
    print('total data:',data_count)
    
    save_path_full = save_path+f"_{resolution}_{data_count}"
    
    valid_count = _save_to_lmdb(save_path_full, data_items_list)
    
    if valid_count != data_count:
        save_path_full_correct = save_path+f"_{resolution}_{valid_count}"
        cmd = f"mv {save_path_full} {save_path_full_correct}"
        subprocess.run([cmd], shell=True, check=True)

if '__main__' == __name__:
    main()

