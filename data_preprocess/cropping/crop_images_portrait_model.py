import os
import numpy as np
from PIL import Image
import argparse
import sys
import torch
from omegaconf import OmegaConf
from cropping.one_euro_filter import SmootherHighdim, Smoother2d

s_smoother = SmootherHighdim(min_cutoff = 0.0001, beta = 0.005)
t_smoother = Smoother2d(min_cutoff = 0.001, beta = 0.05)

# transfer 68 landmarks to 5 landmarks
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# calculating least squres problem between 3D landmarks and 2D landmarks for image alignment
def POS(xp,x,cate=None):
    npts = xp.shape[0]

    A = np.zeros([2*npts,8])
    A[0:2*npts-1:2,0:3] = x
    # A[0:2*npts-1:2,2] = 0 # new add
    A[0:2*npts-1:2,3] = 1
    A[1:2*npts:2,4:7] = x
    # A[1:2*npts:2,6] = 0 # new add
    A[1:2*npts:2,7] = 1
    b = np.reshape(xp,[2*npts,1])

    weight = 1

    A = A * weight
    b = b * weight

    k,_,_,_ = np.linalg.lstsq(A,b)

    R1 = k[0:3].squeeze()
    R2 = k[4:7].squeeze()
    sTx = k[3]
    sTy = k[7]

    cz = np.cross(R1, R2)
    y = np.array([0, 1, 0])
    cx = np.cross(y, cz)
    cy = np.cross(cz, cx)
    cx = cx / np.linalg.norm(cx)
    cy = cy / np.linalg.norm(cy)
    cz = cz / np.linalg.norm(cz)

    yaw = np.arctan2(-cz[0], cz[2]) + 0.5 * np.pi
    pitch = np.arctan(-cz[1] / np.linalg.norm(cz[::2])) + 0.5 * np.pi
    roll1 = (np.sign(np.dot(cz, np.cross(cx, R1))) * np.arccos(np.dot(R1, cx) / np.linalg.norm(R1)) + np.sign(np.dot(cz, np.cross(cy, R2))) * np.arccos(np.dot(R2, cy) / np.linalg.norm(R2))) / 2
    roll2 = np.arctan2(-xp[1, 1] + xp[0, 1], xp[1, 0] - xp[0, 0])
    roll = roll2 + np.sign(roll1 - roll2) * np.log(np.abs(roll1 - roll2)/np.pi*180)*np.pi/180

    scale = 0.5 * np.linalg.norm(R1) + 0.5 * np.linalg.norm(R2)

    translate = np.stack([sTx, sTy],axis = 0)

    return yaw, pitch, roll, translate, scale


def resize_n_crop_img(img, lm, ldmk, ldmk_3d, t, s, target_size=256.):
    w0,h0 = img.size
    # print(w0, h0)

    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    img = img.resize((w,h),resample = Image.LANCZOS)
    # print(w, h)

    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = (left + target_size).astype(np.int32)
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = (up + target_size).astype(np.int32)

    padding_len = max([abs(min(0,left)),abs(min(0,up)),max(right-w,0),max(below-h,0)])
    if padding_len > 0:
        img = np.array(img)
        # img = np.pad(img,pad_width=((padding_len,padding_len),(padding_len,padding_len),(0,0)),mode='reflect')
        img = np.pad(img,pad_width=((padding_len,padding_len),(padding_len,padding_len),(0,0)),mode='edge')
        img = Image.fromarray(img)

    crop_img = img.crop((left+padding_len,up+padding_len,right+padding_len,below+padding_len))

    crop_lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                    t[1] + h0/2], axis=1)*s
    crop_lm = crop_lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])


    ldmk = np.stack([ldmk[:, 0] - t[0] + w0/2, ldmk[:, 1] -
                    t[1] + h0/2], axis=1)*s
    ldmk = ldmk - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])


    ldmk_3d = np.stack([ldmk_3d[:, 0] - t[0] + w0/2, ldmk_3d[:, 1] -
                    t[1] + h0/2], axis=1)*s
    ldmk_3d = ldmk_3d - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return crop_img, crop_lm, ldmk, ldmk_3d


def align_img_bfm(img, lm, ldmk, ldmk_3d, target_size=224., rescale_factor=102., index=-1, use_smooth=False):
    
    # standard 5 facial landmarks
    lm3D = np.array([
        [-0.31148657,  0.09036078,  0.13377953],
        [ 0.30979887,  0.08972035,  0.13179526],
        [ 0.0032535 , -0.24617933,  0.55244243],
        [-0.25216928, -0.5813392 ,  0.22405732],
        [ 0.2484662 , -0.5812824 ,  0.22235769],
    ])

    lm3D[:,2] += 0.4
    lm3D[:,1] += 0.1

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm
        
    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    _, _, _, t, s = POS(lm5p, lm3D)
    s = rescale_factor/s

    if index == 0:
        s_smoother.reset()
        t_smoother.reset()
    
    if use_smooth:
        t = t_smoother.smooth(index, t)
        s = s_smoother.smooth(index, s)
    
    # processing the image
    img_new, lm_new, ldmk, ldmk_3d = resize_n_crop_img(img, lm, ldmk, ldmk_3d, t, s, target_size=target_size)
    trans_params = np.array([w0, h0, s, t[0][0], t[1][0]])

    return trans_params, img_new, lm_new, ldmk, ldmk_3d

        