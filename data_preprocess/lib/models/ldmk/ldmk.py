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
import torchvision.transforms as transforms
from PIL import Image
import math

##### self defined package
from lib.models.ldmk.hrnet import LandmarkDetector

gauss_kernel = None

class ldmkDetector(nn.Module):
    def __init__(self, cfg):
        super(ldmkDetector, self).__init__()
        if cfg.model.ldmk.model_name == "h3r":
            self.model = LandmarkDetector(cfg.model.ldmk.model_path)
        else:
            print("Error: the model {} of landmark is not exists".format(cfg.model.ldmk.model_name))

        self.model.eval()
        self.model.cuda()
        
        self.size = cfg.model.ldmk.img_size # 256
        
        self.landmark_transform = transforms.Compose([
            transforms.Resize(size=(self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _transform(self, img):
        
        h, w, c = img.shape
        img = img[:, :, ::-1]
        img = Image.fromarray(img.astype(np.uint8))
        img = self.landmark_transform(img)
        img = img.type(torch.FloatTensor).unsqueeze(0)
        
        
        img = img.cuda()
        
        return img, h, w
        
    def forward(self, img):
        img, h, w  = self._transform(img)
        
        _, landmarks = self.model(img)
        
        landmarks = landmarks / torch.Tensor([self.size / w, self.size / h]).reshape(1, 1, 2).cuda()
        
        landmarks = landmarks.detach().cpu().numpy()
        
        return landmarks


class ldmk3dDetector(nn.Module):
    def __init__(self, cfg):
        super(ldmk3dDetector, self).__init__()
        self.model_3d = torch.jit.load(cfg.model.ldmk_3d.model_path)
        self.model_depth = torch.jit.load(cfg.model.ldmk_3d.model_depth_path)

        self.model_3d.eval()
        self.model_depth.eval()
        self.model_3d.cuda()
        self.model_depth.cuda()
        
        self.size = cfg.model.ldmk.img_size # 256
        
        self.landmark_transform = transforms.Compose([
            transforms.Resize(size=(self.size, self.size)),
            transforms.ToTensor(),
        ])
        
    def _transform(self, img):
        
        h, w, c = img.shape
        img = img[:, :, ::-1]
        img = Image.fromarray(img.astype(np.uint8))
        img = self.landmark_transform(img)
        img = img.type(torch.FloatTensor).unsqueeze(0)
        
        img = img.cuda()
        
        return img, h, w

    def get_cropped_img(self, img, box):
        center = torch.tensor(
            [box[2] - (box[2] - box[0]) / 2.0, box[3] - (box[3] - box[1]) / 2.0])
        center[1] = center[1] - (box[3] - box[1]) * 0.12
        scale = (box[2] - box[0] + box[3] - box[1]) / 192

        inp = crop(img, center, scale)
        return inp, center, scale
        
    def forward(self, img, boxes):
        ldmks = []
        for box in boxes:
            img_cropped, center, scale = self.get_cropped_img(img, box)
            img_cropped, h, w  = self._transform(img_cropped)
            
            out = self.model_3d(img_cropped).detach()
            out = out.cpu().numpy()
            
            pts, pts_img, scores = get_preds_fromhm(out, center.numpy(), scale)
            pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            scores = scores.squeeze(0)

            heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
            for i in range(68):
                if pts[i, 0] > 0 and pts[i, 1] > 0:
                    heatmaps[i] = draw_gaussian(
                        heatmaps[i], pts[i], 2)
            heatmaps = torch.from_numpy(
                heatmaps).unsqueeze_(0)

            heatmaps = heatmaps.cuda()
            depth_pred = self.model_depth(
                torch.cat((img_cropped, heatmaps), 1)).data.cpu().view(68, 1)
            pts_img = torch.cat(
                (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1).detach().cpu().numpy()
            
            ldmks.append(pts_img)
        return np.array(ldmks)

def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.
    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    hm_reshape = hm.reshape(B, C, H * W)
    idx = np.argmax(hm_reshape, axis=-1)
    scores = np.take_along_axis(hm_reshape, np.expand_dims(idx, axis=-1), axis=-1).squeeze(-1)
    preds, preds_orig = _get_preds_fromhm(hm, idx, center, scale)

    return preds, preds_orig, scores

def _get_preds_fromhm(hm, idx, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps and the
    coresponding locations of the maximums. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.
    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    idx += 1
    preds = idx.repeat(2).reshape(B, C, 2).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
    preds[:, :, 1] = np.floor((preds[:, :, 1] - 1) / H) + 1

    for i in range(B):
        for j in range(C):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j] += np.sign(diff) * 0.25

    preds -= 0.5

    preds_orig = np.zeros_like(preds)
    if center is not None and scale is not None:
        for i in range(B):
            for j in range(C):
                preds_orig[i, j] = transform_np(
                    preds[i, j], center, scale, H, True)

    return preds, preds_orig

def draw_gaussian(image, point, sigma):
    global gauss_kernel
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    if gauss_kernel is None:
        g = _gaussian(size)
        gauss_kernel = g
    else:
        g = gauss_kernel
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image

def crop(image, center, scale, resolution=256.0):
    """Center crops an image or set of heatmaps
    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face
    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})
    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg

def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.
    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.
    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution
    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()

def transform_np(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.
    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.
    Arguments:
        point {numpy.array} -- the input 2D point
        center {numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution
    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.ascontiguousarray(np.linalg.pinv(t))

    new_point = np.dot(t, _pt)[0:2]

    return new_point.astype(np.int32)

def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss