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
import torch
import torch.nn as nn
import numpy as np

##### self defined package
from lib.models.fd.models.retinaface import RetinaFace
from lib.models.fd.layers.functions.prior_box import PriorBox
from lib.models.fd.utils.box_utils import decode, decode_landm
from lib.models.fd.utils.nms.py_cpu_nms import py_cpu_nms
# from lib.models.fd.config import cfg_re50 as cfg


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class faceDetector(nn.Module):
    def __init__(self, cfg):
        super(faceDetector, self).__init__()
        
        if cfg.model.fd.model_name == "retinaface":
            self.model = RetinaFace(cfg=cfg.model.fd.config, phase='test')
            self.model = load_model(self.model, cfg.model.fd.model_path, False)
        else:
            print("Error: the model {} of face detect is not exists".format(cfg.model.ldmk.model_name))
            
        self.model.eval()
        self.model.cuda()
        
        self.resize_h = cfg.model.fd.img_size
        self.confidence_threshold = cfg.model.fd.thres # 0.8
        self.nms_threshold = cfg.model.fd.nms_thres # 0.4
        
        self.cfg = cfg.model.fd.config
        
    def _transform(self, img):
        h, w, c = img.shape
        ratio = self.resize_h / h
        
        img = cv2.resize(img, None, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        
        img = img.astype(np.float32)
        img -= (104., 117., 123.)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()
        
        return img, ratio
        
    def forward(self, img):
        img, ratio= self._transform(img)
        
        loc, conf, ldmks = self.model(img)
        
        _, _, h, w = img.shape
        priorbox = PriorBox(self.cfg, image_size=(h, w))
        priors = priorbox.forward()
        priors = priors.cuda()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        
        boxes = boxes * torch.Tensor([w, h, w, h]).cuda() / ratio
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        
        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        
        return dets