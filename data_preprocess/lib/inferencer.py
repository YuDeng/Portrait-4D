#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

##### python internal and external package
import os
import cv2
import time
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import skimage.transform as transform
from PIL import Image
import copy

##### self defined package
from lib.model_builder import make_model
from lib.face_detect_ldmk_pipeline import FaceLdmkDetector
from cropping.crop_images_portrait_model import align_img_bfm
import torch.nn.functional as F


def parsing_log(log_path):
    with open(log_path, 'r') as f:
        info_log = f.readlines()
        
    for item in info_log:
        if "Duration" in item:
            _time = item.split(',')[0][-11:]
            
    hh = _time[:2]
    mm = _time[3:5]
    ss = _time[6:8]
    ms = _time[9:]
    
    s = int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 100
    
    return _time, s

def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60 
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                    int(sec), int(end))

def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)

def _fix_image( image):
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

class Tester(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.img_size = self.cfg.img_size
        self.batch_size = 8
        self.resolution_inp = 224
        
        self.net_recon, self.net_fd, self.net_ldmk, self.net_ldmk_3d = make_model(self.cfg)
        self.net_recon.setup(self.cfg.model.facerecon)

        self.net_recon.device = torch.device('cuda')  # net_recon does not inherit torch.nn.module class
        self.net_recon.net_recon.cuda()
        self.net_fd.cuda()
        self.net_ldmk.cuda()
        self.net_ldmk_3d.cuda()

        ### set eval and freeze models
        self.net_recon.eval()
        self.net_fd.eval()
        self.net_ldmk.eval()
        self.net_ldmk_3d.eval()
    
        self.fd_ldmk_detector = FaceLdmkDetector(self.net_fd, self.net_ldmk, self.net_ldmk_3d)
        self.stand_index = np.array([96, 97, 54, 76, 82])
        
        self.target_size = 512. # target image size after cropping
        self.rescale_factor = 180 # a scaling factor determining the size of the head in the image, ffhq crop size seems as eg3d crop
        
            
    def inference(self, input_dir, save_dir, video=True, use_crop_smooth=False):

        test_start = time.time()
        
        align_dir = os.path.join(save_dir, "align_images")
        save_dir_bfm = os.path.join(save_dir, "bfm_params")
        save_dir_bfm_vis = os.path.join(save_dir, "bfm_vis")
        save_dir_ldmk2d = os.path.join(save_dir, "2dldmks_align")
        save_dir_ldmk3d = os.path.join(save_dir, "3dldmks_align")
        os.makedirs(align_dir, exist_ok=True)
        os.makedirs(save_dir_bfm, exist_ok=True)
        os.makedirs(save_dir_bfm_vis, exist_ok=True)
        os.makedirs(save_dir_ldmk2d, exist_ok=True)
        os.makedirs(save_dir_ldmk3d, exist_ok=True)
        
        image_list = list(os.listdir(input_dir))
        image_list = [i for i in image_list if i.endswith('.png') or i.endswith('.jpg')]
        image_list.sort()
        
        frame_num = len(image_list)
        print("total {} images in {}".format(frame_num, input_dir))

        if frame_num == 0:
            print("{} images in {}, pass".format(frame_num, input_dir))
            return
        
        flag = False
        count = 0

        s = None
        t = None

        self.fd_ldmk_detector.reset()
        
        with torch.no_grad():
            for img_idx, img_item in tqdm(enumerate(image_list)):
                img_path = os.path.join(input_dir, img_item)
                img = cv2.imread(img_path)
                ih, iw, c = img.shape
                
                try:
                    ldmks, ldmks_3d, boxes = self.fd_ldmk_detector.inference(img)
                except Exception as e:
                    self.fd_ldmk_detector.reset()
                    print(e)
                    print("1")
                    return
                if not video:
                    self.fd_ldmk_detector.reset()
                
                ldmks = ldmks[:1] # only select first detected head
                ldmks_3d = ldmks_3d[:1]
                
                for i in range(len(ldmks)):
                    
                    
                    # extract bfm params via Deep3DRecon
                    input_recon = {'imgs':Image.fromarray(img[:,:,::-1]),'lms':ldmks[i]}
                    bfm_params, trans_params = self.net_recon.forward(input_recon)
                    self.net_recon.compute_visuals()
                    visual = self.net_recon.output_vis.squeeze(0)
                    
                    # reverse y axis for later image cropping process
                    ldmk = ldmks[i].copy()
                    ldmk[:, -1] = ih - 1 - ldmk[:, -1]

                    ldmk_3d = ldmks_3d[i].copy()
                    ldmk_3d[:, 1] = ih - 1 - ldmk_3d[:, 1]

                    ldmk_2d_5pt = ldmks[i, self.stand_index]
                    ldmk_2d_5pt[:, -1] = ih - 1 - ldmk_2d_5pt[:, -1]
                    
                    # cropping
                    try:
                        trans_params, im_crop, ldmk_2d_5pt_crop, ldmk, ldmk_3d = align_img_bfm(Image.fromarray(img[:,:,::-1]), ldmk_2d_5pt, ldmk, ldmk_3d, target_size=self.target_size, rescale_factor=self.rescale_factor, index=img_idx, use_smooth=use_crop_smooth)
                    except Exception as e:
                        self.fd_ldmk_detector.reset()
                        print(e)
                        print("2")
                        return
                    
                    # reverse back to original y direction
                    ldmk = np.concatenate([ldmk[:, 0:1], self.target_size-ldmk[:, 1:2]], 1)
                    ldmk_3d = np.concatenate([ldmk_3d[:, 0:1], self.target_size-ldmk_3d[:, 1:2]], 1)   
                    
                    # save results                            
                    np.save(os.path.join(save_dir_ldmk2d, img_item.replace(".png", ".npy").replace(".jpg", ".npy")), ldmk)
                    np.save(os.path.join(save_dir_ldmk3d, img_item.replace(".png", ".npy").replace(".jpg", ".npy")), ldmk_3d)
                    np.save(os.path.join(save_dir_bfm, img_item.replace(".png", ".npy").replace(".jpg", ".npy")), bfm_params)
                    im_crop.save(os.path.join(align_dir, img_item), quality=95)
                    Image.fromarray(visual.astype(np.uint8)).save(os.path.join(save_dir_bfm_vis, img_item), quality=95)
        
        test_end = time.time()
        
        print("process time is: {}".format(test_end - test_start))
        self.fd_ldmk_detector.reset()
            
            
if "__main__" == __name__:
    tester = Tester()