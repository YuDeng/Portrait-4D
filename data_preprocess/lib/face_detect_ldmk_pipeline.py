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


class FaceLdmkDetector(object):
    def __init__(self, facedetector, ldmkdetector, ldmk3ddetector):
        self.facedetector = facedetector
        self.ldmkdetector = ldmkdetector
        self.ldmk3ddetector = ldmk3ddetector
        
        self.last_frame = None
        self.frame_count = 0
        self.frame_margin = 15
        
    def reset(self):
        self.last_frame = None
        self.frame_count = 0
    
    def get_box_from_ldmk(self, ldmks):
        boxes_return = []
        for ldmk in ldmks:
            xmin = np.min(ldmk[:, 0])
            xmax = np.max(ldmk[:, 0])
            ymin = np.min(ldmk[:, 1])
            ymax = np.max(ldmk[:, 1])
            
            boxes_return.append([xmin, ymin, xmax, ymax])
            
        return np.array(boxes_return)
    
    def extend_box(self, boxes, ratio=1.5):
        boxes_return = []
        for box in boxes:
            xmin = box[0]; ymin = box[1]
            xmax = box[2]; ymax = box[3]
            
            center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
            h = (ymax - ymin + 1)
            w = (xmax - xmin + 1)
            
            size = np.sqrt(h * w)
            
            extend_size = size * ratio
            
            xmine = center[0] - extend_size / 2
            xmaxe = center[0] + extend_size / 2
            ymine = center[1] - extend_size / 2
            ymaxe = center[1] + extend_size / 2
            
            boxes_return.append([xmine, ymine, xmaxe, ymaxe])
            
        return np.array(boxes_return)
    
    def ldmk_detect(self, img, boxes):
        ldmks = []
        h, w, c = img.shape
        for box in boxes:
            xmin = int(box[0]); ymin = int(box[1])
            xmax = int(box[2]); ymax = int(box[3])

            img_crop = np.zeros([ymax-ymin, xmax-xmin, 3])

            img_crop[
                0-min(ymin, 0): (ymax-ymin) - (max(ymax, h) - h),
                0-min(xmin, 0): (xmax-xmin) - (max(xmax, w) - w),
            ] = img[
                max(ymin, 0):min(ymax, h), 
                max(xmin, 0):min(xmax, w)
            ]
            
            # img_crop = img[
            #     max(ymin, 0):min(ymax, h), 
            #     max(xmin, 0):min(xmax, w)
            # ]
            
            ldmk = self.ldmkdetector(img_crop)
            ldmk = ldmk[0] + np.array([xmin, ymin]).reshape(1, 2)
            ldmks.append(ldmk)
        ldmks = np.array(ldmks)
        ldmks_smooth = []
        for idx in range(len(ldmks)):
            if idx == 0 or idx == len(ldmks) - 1:
                ldmks_smooth.append(ldmks[idx])
            else:
                ldmks_smooth.append(ldmks[idx-1:idx+2].mean(0))
        return np.array(ldmks_smooth)
            
    
    def inference(self, img):
        h, w, c = img.shape
        if self.last_frame is None and self.frame_count % self.frame_margin == 0:
            boxes_scores = self.facedetector(img)
            if len(boxes_scores) == 0:
                return None
            boxes = boxes_scores[:, :4]
            scores = boxes_scores[:, 4]

            boxes_size = ((boxes[:, 3] - boxes[:, 1]) + (boxes[:, 2] - boxes[:, 0])) / 2
            boxes_to_center = ((boxes[:, 3] + boxes[:, 1])/2 - 255)**2 + ((boxes[:, 2] + boxes[:, 0])/2 - 255)**2
            # index = np.argmin(boxes_to_center)
            # print(scores)
            # print(boxes_size)
            index = np.argmax(boxes_size)
            # index = np.argmax(scores)
            boxes = boxes[index:index+1]
            # if boxes_size[index] < h / 6:
            #     return None
            boxes_extend = self.extend_box(boxes, ratio=1.5)
            
            self.frame_count += 1
            
            
        else:
            boxes = self.get_box_from_ldmk(self.last_frame)
            boxes_size = ((boxes[:, 3] - boxes[:, 1]) + (boxes[:, 2] - boxes[:, 0])) / 2
            boxes_center = np.concatenate([((boxes[:, 2] + boxes[:, 0])) / 2, ((boxes[:, 3] + boxes[:, 1])) / 2], 0)
            # if boxes_size < h / 6 or boxes_center[0] < 0 or boxes_center[0] > w or boxes_center[1] < 0 or boxes_center[1] > h:
            #     return None
            boxes_extend = self.extend_box(boxes, ratio=1.8)
            
            self.frame_count += 1
            
        ldmks = self.ldmk_detect(img, boxes_extend)
        ldmks_3d = self.ldmk3ddetector(img, boxes)

        if self.last_frame is None:
            boxes = self.get_box_from_ldmk(ldmks)
        
        self.last_frame = ldmks
        
        return ldmks, ldmks_3d, boxes
        
        
    
    