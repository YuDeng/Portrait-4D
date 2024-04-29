import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../portrait4d/')))
from portrait4d.training.dataloader.protocols import datum_portrait_vfhq_pb2 as datum_pb2

import cv2
import numpy as np
from tqdm import tqdm
import lmdb
import json
import torch
import subprocess
from PIL import Image
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

def valid_label(label):
    extrinsics = torch.from_numpy(label[:16]).reshape(-1,4,4)
    inverse_extrinsics = cam_world_matrix_transform(extrinsics)
    inverse_trans = inverse_extrinsics[:,:3,3]*3
    inverse_trans_z = torch.abs(inverse_trans[:,2]).unsqueeze(0)
    if inverse_trans_z > 10:
        return False
    else:
        return True


def array_to_datum(images, segs, labels, mots):
    assert len(images) == len(segs)
    assert len(images) == len(labels)
    assert len(images) == len(mots)
    datum = datum_pb2.Datum_vfhq()
    datum.width, datum.height, datum.channels = 512, 512, 3
    datum.num = len(images)

    datum.images = bytes(images)
    datum.segs = bytes(segs)
    datum.labels = bytes(labels)
    datum.mots = bytes(mots)

    return datum


def _save_to_lmdb(save_path, folder_list, rescale_camera=False):
    db = lmdb.open(save_path, map_size=1024 ** 4 * 5)
    txn = db.begin(write=True)
    count = 0
    for folder in tqdm(folder_list):
        if not os.path.isfile(os.path.join(folder,'select_index_50.txt')):
            select_idx = list(range(50))
        else:
            select_idx = np.loadtxt(os.path.join(folder,'select_index_50.txt')).reshape(-1).astype(np.int32)
            select_idx = sorted(select_idx)
        if len(select_idx) != 50:
            print('less than 50')
            continue
        img_paths = sorted(os.listdir(os.path.join(folder, 'align_images')))
        if select_idx[-1] >= len(img_paths):
            print('invalid select idx')
            continue            
        
        img_paths = [img_paths[idx] for idx in select_idx]
        img_paths = [os.path.join(folder, 'align_images', f) for f in img_paths]
        seg_paths = [f.replace('align_images','segs') for f in img_paths]
        label_paths = [f.replace('align_images','flame_optim_params').replace('.png','.npy').replace('.jpg','.npy') for f in img_paths]
        mot_paths = [f.replace('align_images','motion_feats').replace('.png','.npy').replace('.jpg','.npy') for f in img_paths]
        
        img_list = []
        seg_list = []
        label_list = []
        mot_list = []

        for img_path, seg_path, label_path, mot_path in zip(img_paths,seg_paths,label_paths,mot_paths):

            if not os.path.isfile(label_path):
                print('no label')
                continue

            if not os.path.isfile(mot_path):
                print('no motion embedding')
                continue

            label = np.load(label_path)

            # if not valid_label(label):
            #     print('invalid label')
            #     continue
            
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
            
            img = np.array(Image.open(img_path))
            seg = np.array(Image.open(seg_path))

            img_list.append(img)
            seg_list.append(seg)
            label_list.append(label)
            mot_list.append(mot_select)
        
        if len(img_list) == 0:
            print('no image')
            continue
        
        img_list = np.stack(img_list,0)
        seg_list = np.stack(seg_list,0)
        label_list = np.stack(label_list,0)
        mot_list = np.stack(mot_list,0)

        datum = array_to_datum(
            img_list, seg_list, label_list, mot_list)
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
    
    save_path = os.path.join(output_dir, "VFHQ_sub50")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    folder_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if 'align_images' in os.listdir(os.path.join(data_dir,f))\
        and 'flame_optim_params' in os.listdir(os.path.join(data_dir,f)) and 'motion_feats' in os.listdir(os.path.join(data_dir,f)) and 'segs' in os.listdir(os.path.join(data_dir,f))]

    folder_list = [f for f in folder_list if len(os.listdir(os.path.join(f,'flame_optim_params'))) == len(os.listdir(os.path.join(f,'align_images')))]
    folder_list = [f for f in folder_list if len(os.listdir(os.path.join(f,'segs'))) == len(os.listdir(os.path.join(f,'align_images')))]
    folder_list = [f for f in folder_list if len(os.listdir(os.path.join(f,'motion_feats'))) == len(os.listdir(os.path.join(f,'align_images')))]
    
    folder_list = sorted(folder_list)
    if not max_len == -1:
        folder_list = folder_list[:max_len]
    data_count = len(folder_list)
    print('total folder:',data_count)
    
    save_path_full = save_path+f"_{resolution}_{data_count}"
    valid_count = _save_to_lmdb(save_path_full, folder_list)

    if valid_count != data_count:
        save_path_full_correct = save_path+f"_{resolution}_{valid_count}"
        cmd = f"mv {save_path_full} {save_path_full_correct}"
        subprocess.run([cmd], shell=True, check=True)
    
if '__main__' == __name__:
    main()