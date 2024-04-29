# Dataloader for training Portrait4D, modified from EG3D: https://github.com/NVlabs/eg3d

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import sys
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from training.dataloader.protocols import datum_portrait_genhead_pb2 as datum_pb2_genhead
from training.dataloader.protocols import datum_portrait_genhead_static_pb2 as datum_pb2_genhead_static
from training.dataloader.protocols import datum_portrait_genhead_new_pb2 as datum_pb2_genhead_new
from training.dataloader.protocols import datum_portrait_genhead_static_new_pb2 as datum_pb2_genhead_static_new
from training.dataloader.protocols import datum_portrait_vfhq_pb2 as datum_pb2_vfhq
from training.dataloader.protocols import datum_portrait_ffhq_pb2 as datum_pb2_ffhq
import lmdb
import cv2
from scipy.spatial.transform import Rotation as R
# try:
#     import pyspng
# except ImportError:
pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataloader for real video set
class OneShotReconSegLmdbFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to datalist.
        resolution      = None, # Ensure specific resolution, None = highest available.
        data_type      = "vfhq",# Deprecated
        static         = False, # if True, only select multiview static frames instead of frames with different motions
        use_flame_mot  = False, # whether or not to use FLAME parameters as motion embedding
        max_num = None,
        rescale_camera = False, # Rescale camera extrinsics and intrinscs to align with an older version of camera labels
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        print(self._path)
        self._resolution = resolution
        self._zipfile = None
        self._data_type = data_type
        self.static = static
        self.use_flame_mot = use_flame_mot
        self.rescale_camera = rescale_camera
        
        # initialize lmdb
        if os.path.isdir(self._path):
            self.db = None
            self.txn = None
            self.num = None
            self.datum = None
        else:
            raise IOError('Path must point to a directory or zip')

        img_size = int(self._path.split("/")[-2].split("_")[-2])
        num = int(self._path.split("/")[-2].split("_")[-1])
        img_shape = [3, img_size, img_size]
        raw_shape = [num] + img_shape
        if max_num is not None:
            raw_shape = [max_num] + img_shape
        if resolution is None:
            self._resolution = raw_shape[2]
        name = os.path.splitext(os.path.basename(self._path))[0]
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def open_lmdb(self):
        self.db = lmdb.open(self._path, map_size=1024 ** 4, readonly=True, lock=False)
        self.txn = self.db.begin()
        self.num = int(self.txn.get('num_samples'.encode()))
        self.datum = datum_pb2_vfhq.Datum_vfhq()

    def get_details(self, idx):
        return None

    def get_label_std(self):
        return 0

    @property
    def resolution(self):
        return self._resolution

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels, _ = self._load_raw_labels(0,[0,0])
            self._label_shape = raw_labels.shape
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        images = np.fromstring(self.datum.images, dtype=np.uint8).reshape(self.datum.num,-1)
        frame_idx = np.random.randint(self.datum.num,size=2)

        source_image = images[frame_idx[0]]
        driving_image = images[frame_idx[1]]
        source_image = source_image.reshape(self.datum.height,self.datum.width,self.datum.channels)
        driving_image = driving_image.reshape(self.datum.height,self.datum.width,self.datum.channels)
        
        if source_image.ndim == 2:
            source_image = source_image[:, :, np.newaxis] # HW => HWC
        if driving_image.ndim == 2:
            driving_image = driving_image[:, :, np.newaxis] # HW => HWC

        source_image = source_image.transpose(2, 0, 1) # HWC => CHW
        driving_image = driving_image.transpose(2, 0, 1) # HWC => CHW
        
        return source_image, driving_image, frame_idx

    def _load_raw_seg(self, raw_idx, frame_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        seg = np.fromstring(self.datum.segs, dtype=np.uint8).reshape(self.datum.num,-1)
        
        source_seg = seg[frame_idx[0]]
        driving_seg = seg[frame_idx[1]]
        source_seg = source_seg.reshape(self.datum.height,self.datum.width,-1)
        driving_seg = driving_seg.reshape(self.datum.height,self.datum.width,-1)
        
        if source_seg.ndim == 2:
            source_seg = source_seg[:, :, np.newaxis] # HW => HWC
        if driving_seg.ndim == 2:
            driving_seg = driving_seg[:, :, np.newaxis] # HW => HWC
        source_seg = source_seg.transpose(2, 0, 1) # HWC => CHW
        driving_seg = driving_seg.transpose(2, 0, 1) # HWC => CHW

        if source_seg.shape[0] == 1:
            source_seg = np.tile(source_seg, (3, 1, 1))

        if driving_seg.shape[0] == 1:
            driving_seg = np.tile(driving_seg, (3, 1, 1))
        return source_seg, driving_seg

    def _load_raw_labels(self, raw_idx, frame_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        labels = np.fromstring(self.datum.labels, dtype=np.float32).reshape(self.datum.num,-1)

        # Use average yaw and roll angles across all frames in a video clip as the camera pose to rectify extracted head pose
        head_poses = labels[:,425:428]
        compose_rot = R.from_rotvec(head_poses)
        compose_euler = compose_rot.as_euler('xyz')
        compose_euler_mean = np.mean(compose_euler,axis=0,keepdims=True)
        cam_euler = np.stack([np.zeros_like(compose_euler_mean[...,0]), compose_euler_mean[...,1], compose_euler_mean[...,2]], axis=-1)
        cam_rot = R.from_euler('xyz', cam_euler).as_matrix()
        cam_rot_inverse = np.transpose(cam_rot,[0,2,1])
        head_rot_rectified = np.matmul(cam_rot_inverse,compose_rot.as_matrix())
        head_poses = R.from_matrix(head_rot_rectified).as_rotvec()
        labels[:,425:428] = head_poses

        source_labels = labels[frame_idx[0]]
        driving_labels = labels[frame_idx[1]]
        
        if self.rescale_camera:
            if frame_idx[0] == frame_idx[1]:
                intrinsics = source_labels[16:25].reshape(3,3)
                
                # normalize intrinsics
                if self._resolution != intrinsics[0,2]*2:
                    intrinsics[:2,:] *= (0.5*self._resolution/intrinsics[0,2])

                intrinsics[0, 0] /= self._resolution
                intrinsics[1, 1] /= self._resolution
                intrinsics[0, 2] /= self._resolution
                intrinsics[1, 2] /= self._resolution       
                
                # rescale extrinsics
                extrinsics = source_labels[:16].reshape(4,4) # Our face scale is around 0.1~0.2. Multiply by 3 to match the scale of EG3D
                extrinsics[:3,3] *= 3
            else:    
                for labels in [source_labels, driving_labels]:
                    intrinsics = labels[16:25].reshape(3,3)
                    if self._resolution != intrinsics[0,2]*2:
                        intrinsics[:2,:] *= (0.5*self._resolution/intrinsics[0,2])

                    intrinsics[0, 0] /= self._resolution
                    intrinsics[1, 1] /= self._resolution
                    intrinsics[0, 2] /= self._resolution
                    intrinsics[1, 2] /= self._resolution       

                    extrinsics = labels[:16].reshape(4,4)  
                    extrinsics[:3,3] *= 3

        return source_labels, driving_labels

    def _load_raw_motions(self, raw_idx, frame_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        motions = np.fromstring(self.datum.mots, dtype=np.float32).reshape(self.datum.num,-1)
        motions = motions[:,:548]
        source_motions = motions[frame_idx[0]]
        driving_motions = motions[frame_idx[1]]

        return source_motions, driving_motions

    def get_label(self, raw_labels):
        label = raw_labels[:25]
        return label.copy()

    def get_shape_param(self, raw_labels):
        label = raw_labels[25:325]
        return label.copy()

    def get_exp_param(self, raw_labels):
        label = raw_labels[325:425]
        return label.copy()

    def get_exp_param_w_jaw_pose(self, raw_labels):
        label = np.concatenate([raw_labels[325:425],raw_labels[428:431]],axis=0)
        return label.copy()

    def get_pose_param(self, raw_labels):
        label = raw_labels[425:431]
        return label.copy()

    def get_eye_pose_param(self, raw_labels):
        label = raw_labels[431:437]
        return label.copy()

    def get_label_all(self,raw_labels):
        c = self.get_label(raw_labels)
        shape_param = self.get_shape_param(raw_labels)
        exp_param = self.get_exp_param(raw_labels)
        pose_param = self.get_pose_param(raw_labels)
        eye_pose_param = self.get_eye_pose_param(raw_labels)
        return c, shape_param, exp_param, pose_param, eye_pose_param

    def __getitem__(self, idx):
        source_image, driving_image, frame_idx = self._load_raw_image(self._raw_idx[idx])
        source_seg, driving_seg = self._load_raw_seg(self._raw_idx[idx], frame_idx)
        source_labels, driving_labels = self._load_raw_labels(self._raw_idx[idx], frame_idx)
        source_motions, driving_motions = self._load_raw_motions(self._raw_idx[idx], frame_idx)
        assert isinstance(source_image, np.ndarray)
        assert isinstance(driving_image, np.ndarray)
        assert isinstance(source_seg, np.ndarray)
        assert isinstance(driving_seg, np.ndarray)
        assert list(source_image.shape) == self.image_shape
        assert source_seg.shape[1] == self.image_shape[1] and source_seg.shape[2] == self.image_shape[2]
        assert source_image.dtype == np.uint8
        
        # use flame parameters as motion embedding if use_flame_mot is true
        if self.use_flame_mot:
            source_motions = np.concatenate([self.get_exp_param_w_jaw_pose(source_labels),self.get_eye_pose_param(source_labels)],axis=0)
            driving_motions = np.concatenate([self.get_exp_param_w_jaw_pose(driving_labels),self.get_eye_pose_param(driving_labels)],axis=0)

        return source_image.copy(), driving_image.copy(), driving_image.copy(), driving_seg.copy(), np.zeros([]), np.zeros([]), np.zeros([]), np.zeros([]), np.zeros([]), self.get_label(driving_labels), self.get_shape_param(source_labels), self.get_exp_param(driving_labels), self.get_pose_param(driving_labels), self.get_eye_pose_param(driving_labels), source_motions.copy(), driving_motions.copy()

#----------------------------------------------------------------------------
# Dataloader for real image set
class OneShotReconSegLmdbFolderDataset_Static(OneShotReconSegLmdbFolderDataset):

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._load_raw_labels(0)
            self._label_shape = raw_labels.shape
        return list(self._label_shape)

    def open_lmdb(self):
        self.db = lmdb.open(self._path, map_size=1024 ** 4, readonly=True, lock=False)
        self.txn = self.db.begin()
        self.num = int(self.txn.get('num_samples'.encode()))
        self.datum = datum_pb2_ffhq.Datum_ffhq()

    def _load_raw_image(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        image = np.fromstring(self.datum.images, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]] # bgr -> rgb
        
        image = image.reshape(self.datum.height,self.datum.width,self.datum.channels)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_seg(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        seg = np.fromstring(self.datum.segs, dtype=np.uint8)
        seg = cv2.imdecode(seg, cv2.IMREAD_COLOR)        

        if seg.ndim == 2:
            seg = seg[:, :, np.newaxis] # HW => HWC
        seg = seg.transpose(2, 0, 1) # HWC => CHW

        if seg.shape[0] == 1:
            seg = np.tile(seg, (3, 1, 1))

        return seg

    def _load_raw_labels(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        labels = np.fromstring(self.datum.labels, dtype=np.float32).reshape(-1)
        labels[425:428] = 0 # set head pose to zero and use camera pose instead
        
        if self.rescale_camera:
            intrinsics = labels[16:25].reshape(3,3)
            
            # normalize intrinsics
            if self._resolution != intrinsics[0,2]*2:
                intrinsics[:2,:] *= (0.5*self._resolution/intrinsics[0,2])

            intrinsics[0, 0] /= self._resolution
            intrinsics[1, 1] /= self._resolution
            intrinsics[0, 2] /= self._resolution
            intrinsics[1, 2] /= self._resolution       
            
            # rescale extrinsics
            extrinsics = labels[:16].reshape(4,4) # Our face scale is around 0.1~0.2. Multiply by 3 to match the scale of EG3D
            extrinsics[:3,3] *= 3

        return labels

    def _load_raw_motions(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        motions = np.fromstring(self.datum.mots, dtype=np.float32).reshape(-1)
        motions = motions[:548]

        return motions

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        seg = self._load_raw_seg(self._raw_idx[idx])
        labels = self._load_raw_labels(self._raw_idx[idx])
        motions = self._load_raw_motions(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert isinstance(seg, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert seg.shape[1] == self.image_shape[1] and seg.shape[2] == self.image_shape[2]
        assert image.dtype == np.uint8

        return image.copy(), image.copy(), image.copy(), seg.copy(), np.zeros([]), np.zeros([]), np.zeros([]), np.zeros([]), np.zeros([]), self.get_label(labels), self.get_shape_param(labels), self.get_exp_param(labels), self.get_pose_param(labels), self.get_eye_pose_param(labels), motions.copy(), motions.copy()

#-----------------------------------------------------------------------------
# Dataloader for GenHead generated 4D data (base version)
class GenHeadReconSegLmdbFolderDataset(OneShotReconSegLmdbFolderDataset):

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels, _, _ = self._load_raw_labels(0,[0,0],[0,0,0])
            self._label_shape = raw_labels.shape
        return list(self._label_shape)

    def _load_raw_image(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        images = np.fromstring(self.datum.images, dtype=np.uint8).reshape(self.datum.num//5,5,-1)
        frame_idx = np.random.randint(self.datum.num//5,size=2)
        
        # use same motion for source and driving if static is true
        if self.static:
            frame_idx[1] = frame_idx[0]

        view_idx = np.random.randint(5,size=3)
        source_image = images[frame_idx[0],view_idx[0]]
        driving_image = images[frame_idx[1],view_idx[1]]
        recon_image = images[frame_idx[1],view_idx[2]]

        source_image = source_image.reshape(self.datum.height,self.datum.width,self.datum.channels)
        driving_image = driving_image.reshape(self.datum.height,self.datum.width,self.datum.channels)
        recon_image = recon_image.reshape(self.datum.height,self.datum.width,self.datum.channels)
        if source_image.ndim == 2:
            source_image = source_image[:, :, np.newaxis] # HW => HWC
        if driving_image.ndim == 2:
            driving_image = driving_image[:, :, np.newaxis] # HW => HWC
        if recon_image.ndim == 2:
            recon_image = recon_image[:, :, np.newaxis] # HW => HWC

        source_image = source_image.transpose(2, 0, 1) # HWC => CHW
        driving_image = driving_image.transpose(2, 0, 1) # HWC => CHW
        recon_image = recon_image.transpose(2, 0, 1) # HWC => CHW
        
        return source_image, driving_image, recon_image, frame_idx, view_idx

    def _load_raw_seg(self, raw_idx, frame_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        seg = np.fromstring(self.datum.segs, dtype=np.uint8).reshape(self.datum.num//5,5,-1)
    
        source_seg = seg[frame_idx[0],view_idx[0]]
        driving_seg = seg[frame_idx[1],view_idx[1]]
        recon_seg = seg[frame_idx[1],view_idx[2]]
        source_seg = source_seg.reshape(self.datum.height,self.datum.width,-1)
        driving_seg = driving_seg.reshape(self.datum.height,self.datum.width,-1)
        recon_seg = recon_seg.reshape(self.datum.height,self.datum.width,-1)
        if source_seg.ndim == 2:
            source_seg = source_seg[:, :, np.newaxis] # HW => HWC
        if driving_seg.ndim == 2:
            driving_seg = driving_seg[:, :, np.newaxis] # HW => HWC
        if recon_seg.ndim == 2:
            recon_seg = recon_seg[:, :, np.newaxis] # HW => HWC
        source_seg = source_seg.transpose(2, 0, 1) # HWC => CHW
        driving_seg = driving_seg.transpose(2, 0, 1) # HWC => CHW
        recon_seg = recon_seg.transpose(2, 0, 1) # HWC => CHW

        if source_seg.shape[0] == 1:
            source_seg = np.tile(source_seg, (3, 1, 1))

        if driving_seg.shape[0] == 1:
            driving_seg = np.tile(driving_seg, (3, 1, 1))

        if recon_seg.shape[0] == 1:
            recon_seg = np.tile(recon_seg, (3, 1, 1))

        return source_seg, driving_seg, recon_seg

    def _load_raw_labels(self, raw_idx, frame_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        labels = np.fromstring(self.datum.labels, dtype=np.float32).reshape(self.datum.num//5,5,-1)

        source_labels = labels[frame_idx[0], view_idx[0]]
        driving_labels = labels[frame_idx[1], view_idx[1]]
        recon_labels = labels[frame_idx[1], view_idx[2]]

        return source_labels, driving_labels, recon_labels

    def _load_raw_motions(self, raw_idx, frame_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        motions = np.fromstring(self.datum.mots, dtype=np.float32).reshape(self.datum.num//5,5,-1)
        motions = motions[...,:548]
        source_motions = motions[frame_idx[0], view_idx[0]]
        driving_motions = motions[frame_idx[1], view_idx[1]]
        recon_motios = motions[frame_idx[1], view_idx[2]]

        return source_motions, driving_motions, recon_motios

    def __getitem__(self, idx):
        source_image, driving_image, recon_image, frame_idx, view_idx = self._load_raw_image(self._raw_idx[idx])
        source_seg, driving_seg, recon_seg = self._load_raw_seg(self._raw_idx[idx], frame_idx, view_idx)
        source_labels, driving_labels, recon_labels = self._load_raw_labels(self._raw_idx[idx], frame_idx, view_idx)
        source_motions, driving_motions, recon_motions = self._load_raw_motions(self._raw_idx[idx], frame_idx, view_idx)

        assert isinstance(source_image, np.ndarray)
        assert isinstance(driving_image, np.ndarray)
        assert isinstance(recon_image, np.ndarray)
        assert isinstance(source_seg, np.ndarray)
        assert isinstance(driving_seg, np.ndarray)
        assert isinstance(recon_seg, np.ndarray)
        assert list(source_image.shape) == self.image_shape
        assert source_seg.shape[1] == self.image_shape[1] and source_seg.shape[2] == self.image_shape[2]
        assert source_image.dtype == np.uint8

        return source_image.copy(), driving_image.copy(), recon_image.copy(), recon_seg.copy(), self.get_label(recon_labels), self.get_shape_param(source_labels), self.get_exp_param(driving_labels), self.get_pose_param(driving_labels), self.get_eye_pose_param(driving_labels), driving_motions.copy()

#-----------------------------------------------------------------------------
# Dataloader for GenHead generated 4D data (include depth, feature maps, and triplane features)
class GenHeadReconSegLmdbFolderDatasetV2(GenHeadReconSegLmdbFolderDataset):

    def open_lmdb(self):
        self.db = lmdb.open(self._path, map_size=1024 ** 4, readonly=True, lock=False)
        self.txn = self.db.begin()
        self.num = int(self.txn.get('num_samples'.encode()))
        self.datum = datum_pb2_genhead.Datum()

    def _load_raw_depth(self, raw_idx, frame_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        depth = np.fromstring(self.datum.depths, dtype=np.float16).reshape(self.datum.num//5,5,-1)
    
        source_depth = depth[frame_idx[0],view_idx[0]]
        driving_depth = depth[frame_idx[1],view_idx[1]]
        recon_depth = depth[frame_idx[1],view_idx[2]]
        source_depth = source_depth.reshape(64,64,-1)
        driving_depth = driving_depth.reshape(64,64,-1)
        recon_depth = recon_depth.reshape(64,64,-1)

        source_depth = source_depth.transpose(2, 0, 1) # HWC => CHW
        driving_depth = driving_depth.transpose(2, 0, 1) # HWC => CHW
        recon_depth = recon_depth.transpose(2, 0, 1) # HWC => CHW


        return source_depth, driving_depth, recon_depth

    def _load_raw_feature(self, raw_idx, frame_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        feature = np.fromstring(self.datum.features, dtype=np.float16).reshape(self.datum.num//5,5,-1)
    
        source_feature = feature[frame_idx[0],view_idx[0]]
        driving_feature = feature[frame_idx[1],view_idx[1]]
        recon_feature = feature[frame_idx[1],view_idx[2]]
        
        source_feature = source_feature.reshape(64,64,-1)
        driving_feature = driving_feature.reshape(64,64,-1)
        recon_feature = recon_feature.reshape(64,64,-1)

        source_feature = source_feature.transpose(2, 0, 1) # HWC => CHW
        driving_feature = driving_feature.transpose(2, 0, 1) # HWC => CHW
        recon_feature = recon_feature.transpose(2, 0, 1) # HWC => CHW


        return source_feature, driving_feature, recon_feature

    def _load_raw_triplane(self, raw_idx, frame_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        triplane = np.fromstring(self.datum.triplanes, dtype=np.float16).reshape(self.datum.num//5,-1)
    
        source_triplane = triplane[frame_idx[0]]
        driving_triplane = triplane[frame_idx[1]]
        recon_triplane = triplane[frame_idx[1]]

        source_triplane = source_triplane.reshape(-1,35) # [N,3+32]
        driving_triplane = driving_triplane.reshape(-1,35)
        recon_triplane = recon_triplane.reshape(-1,35)

        return source_triplane, driving_triplane, recon_triplane

    def __getitem__(self, idx):
        source_image, driving_image, recon_image, frame_idx, view_idx = self._load_raw_image(self._raw_idx[idx])
        source_seg, driving_seg, recon_seg = self._load_raw_seg(self._raw_idx[idx], frame_idx, view_idx)
        source_depth, driving_depth, recon_depth = self._load_raw_depth(self._raw_idx[idx], frame_idx, view_idx)
        source_feature, driving_feature, recon_feature = self._load_raw_feature(self._raw_idx[idx], frame_idx, view_idx)
        source_labels, driving_labels, recon_labels = self._load_raw_labels(self._raw_idx[idx], frame_idx, view_idx)
        source_motions, driving_motions, recon_motions = self._load_raw_motions(self._raw_idx[idx], frame_idx, view_idx)
        source_triplane, driving_triplane, recon_triplane = self._load_raw_triplane(self._raw_idx[idx], frame_idx)

        assert isinstance(source_image, np.ndarray)
        assert isinstance(driving_image, np.ndarray)
        assert isinstance(recon_image, np.ndarray)
        assert isinstance(source_seg, np.ndarray)
        assert isinstance(driving_seg, np.ndarray)
        assert isinstance(recon_seg, np.ndarray)
        assert list(source_image.shape) == self.image_shape
        assert source_seg.shape[1] == self.image_shape[1] and source_seg.shape[2] == self.image_shape[2]
        assert source_image.dtype == np.uint8

        return source_image.copy(), driving_image.copy(), recon_image.copy(), recon_seg.copy(), recon_depth.copy(), recon_feature.copy(), recon_triplane.copy(), self.get_label(recon_labels), self.get_shape_param(source_labels), self.get_exp_param(driving_labels), self.get_pose_param(driving_labels), self.get_eye_pose_param(driving_labels), source_motions.copy(), driving_motions.copy()

#----------------------------------------------------------------------------
# Dataloader for GenHead generated static multiview data (include depth, feature maps, and triplane features)
class GenHeadReconSegLmdbFolderDatasetV2_Static(OneShotReconSegLmdbFolderDataset):

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels, _, _ = self._load_raw_labels(0,[0,0,0])
            self._label_shape = raw_labels.shape
        return list(self._label_shape)

    def open_lmdb(self):
        self.db = lmdb.open(self._path, map_size=1024 ** 4, readonly=True, lock=False)
        self.txn = self.db.begin()
        self.num = int(self.txn.get('num_samples'.encode()))
        self.datum = datum_pb2_genhead_static.Datum_static()

    def _load_raw_image(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        images = np.fromstring(self.datum.images, dtype=np.uint8).reshape(self.datum.num,-1)

        view_idx = np.random.randint(self.datum.num,size=3)

        source_image = images[view_idx[0]]
        driving_image = images[view_idx[1]]
        recon_image = images[view_idx[2]]

        source_image = source_image.reshape(self.datum.height,self.datum.width,self.datum.channels)
        driving_image = driving_image.reshape(self.datum.height,self.datum.width,self.datum.channels)
        recon_image = recon_image.reshape(self.datum.height,self.datum.width,self.datum.channels)
        if source_image.ndim == 2:
            source_image = source_image[:, :, np.newaxis] # HW => HWC
        if driving_image.ndim == 2:
            driving_image = driving_image[:, :, np.newaxis] # HW => HWC
        if recon_image.ndim == 2:
            recon_image = recon_image[:, :, np.newaxis] # HW => HWC

        source_image = source_image.transpose(2, 0, 1) # HWC => CHW
        driving_image = driving_image.transpose(2, 0, 1) # HWC => CHW
        recon_image = recon_image.transpose(2, 0, 1) # HWC => CHW
        return source_image, driving_image, recon_image, view_idx

    def _load_raw_seg(self, raw_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        seg = np.fromstring(self.datum.segs, dtype=np.uint8).reshape(self.datum.num,-1)
    
        source_seg = seg[view_idx[0]]
        driving_seg = seg[view_idx[1]]
        recon_seg = seg[view_idx[2]]
        
        source_seg = source_seg.reshape(self.datum.height,self.datum.width,-1)
        driving_seg = driving_seg.reshape(self.datum.height,self.datum.width,-1)
        recon_seg = recon_seg.reshape(self.datum.height,self.datum.width,-1)
        if source_seg.ndim == 2:
            source_seg = source_seg[:, :, np.newaxis] # HW => HWC
        if driving_seg.ndim == 2:
            driving_seg = driving_seg[:, :, np.newaxis] # HW => HWC
        if recon_seg.ndim == 2:
            recon_seg = recon_seg[:, :, np.newaxis] # HW => HWC
        source_seg = source_seg.transpose(2, 0, 1) # HWC => CHW
        driving_seg = driving_seg.transpose(2, 0, 1) # HWC => CHW
        recon_seg = recon_seg.transpose(2, 0, 1) # HWC => CHW

        if source_seg.shape[0] == 1:
            source_seg = np.tile(source_seg, (3, 1, 1))

        if driving_seg.shape[0] == 1:
            driving_seg = np.tile(driving_seg, (3, 1, 1))

        if recon_seg.shape[0] == 1:
            recon_seg = np.tile(recon_seg, (3, 1, 1))

        return source_seg, driving_seg, recon_seg

    def _load_raw_labels(self, raw_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        labels = np.fromstring(self.datum.labels, dtype=np.float32).reshape(self.datum.num,-1)

        source_labels = labels[view_idx[0]]
        driving_labels = labels[view_idx[1]]
        recon_labels = labels[view_idx[2]]

        return source_labels, driving_labels, recon_labels

    def _load_raw_motions(self, raw_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        motions = np.fromstring(self.datum.mots, dtype=np.float32).reshape(self.datum.num,-1)
        motions = motions[...,:548]
        source_motions = motions[view_idx[0]]
        driving_motions = motions[view_idx[1]]
        recon_motios = motions[view_idx[2]]

        return source_motions, driving_motions, recon_motios

    def _load_raw_depth(self, raw_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        depth = np.fromstring(self.datum.depths, dtype=np.float16).reshape(self.datum.num,-1)
    
        source_depth = depth[view_idx[0]]
        driving_depth = depth[view_idx[1]]
        recon_depth = depth[view_idx[2]]

        source_depth = source_depth.reshape(64,64,-1)
        driving_depth = driving_depth.reshape(64,64,-1)
        recon_depth = recon_depth.reshape(64,64,-1)

        source_depth = source_depth.transpose(2, 0, 1) # HWC => CHW
        driving_depth = driving_depth.transpose(2, 0, 1) # HWC => CHW
        recon_depth = recon_depth.transpose(2, 0, 1) # HWC => CHW

        return source_depth, driving_depth, recon_depth

    def _load_raw_feature(self, raw_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        feature = np.fromstring(self.datum.features, dtype=np.float16).reshape(self.datum.num,-1)
    
        source_feature = feature[view_idx[0]]
        driving_feature = feature[view_idx[1]]
        recon_feature = feature[view_idx[2]]

        source_feature = source_feature.reshape(64,64,-1)
        driving_feature = driving_feature.reshape(64,64,-1)
        recon_feature = recon_feature.reshape(64,64,-1)

        source_feature = source_feature.transpose(2, 0, 1) # HWC => CHW
        driving_feature = driving_feature.transpose(2, 0, 1) # HWC => CHW
        recon_feature = recon_feature.transpose(2, 0, 1) # HWC => CHW

        return source_feature, driving_feature, recon_feature

    def _load_raw_triplane(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        triplane = np.fromstring(self.datum.triplanes, dtype=np.float16).reshape(-1)
    
        source_triplane = triplane
        driving_triplane = triplane
        recon_triplane = triplane

        source_triplane = source_triplane.reshape(-1,35) # [N,3+32]
        driving_triplane = driving_triplane.reshape(-1,35)
        recon_triplane = recon_triplane.reshape(-1,35)

        return source_triplane, driving_triplane, recon_triplane

    def __getitem__(self, idx):
        source_image, driving_image, recon_image, view_idx = self._load_raw_image(self._raw_idx[idx])
        source_seg, driving_seg, recon_seg = self._load_raw_seg(self._raw_idx[idx], view_idx)
        source_depth, driving_depth, recon_depth = self._load_raw_depth(self._raw_idx[idx], view_idx)
        source_feature, driving_feature, recon_feature = self._load_raw_feature(self._raw_idx[idx], view_idx)
        source_labels, driving_labels, recon_labels = self._load_raw_labels(self._raw_idx[idx], view_idx)
        source_motions, driving_motions, recon_motions = self._load_raw_motions(self._raw_idx[idx], view_idx)
        source_triplane, driving_triplane, recon_triplane = self._load_raw_triplane(self._raw_idx[idx])

        assert isinstance(source_image, np.ndarray)
        assert isinstance(driving_image, np.ndarray)
        assert isinstance(recon_image, np.ndarray)
        assert isinstance(source_seg, np.ndarray)
        assert isinstance(driving_seg, np.ndarray)
        assert isinstance(recon_seg, np.ndarray)
        assert list(source_image.shape) == self.image_shape
        assert source_seg.shape[1] == self.image_shape[1] and source_seg.shape[2] == self.image_shape[2]
        assert source_image.dtype == np.uint8

        return source_image.copy(), driving_image.copy(), recon_image.copy(), recon_seg.copy(), recon_depth.copy(), recon_feature.copy(), recon_triplane.copy(), self.get_label(recon_labels), self.get_shape_param(source_labels), self.get_exp_param(driving_labels), self.get_pose_param(driving_labels), self.get_eye_pose_param(driving_labels), source_motions.copy(), driving_motions.copy()

#--------------------------------------------------------------------------------------------------------
# Dataloader for GenHead generated 4D data (include depth, feature maps, triplane features, background, and rendered segmentation)
class GenHeadReconSegLmdbFolderDatasetV2New(GenHeadReconSegLmdbFolderDatasetV2):

    def __init__(self,
        path,                   # Path to datalist.
        resolution      = None, # Ensure specific resolution, None = highest available.
        data_type      = "vfhq",# Deprecated
        static         = False, # if True, only select multiview static frames instead of frames with different motions
        use_flame_mot  = False, # whether or not to use FLAME parameters as motion embedding
        **super_kwargs,         # Additional arguments for the Dataset base class.
        ):
        super().__init__(path=path, resolution=resolution, data_type=data_type, static=static, **super_kwargs)
        self.use_flame_mot = use_flame_mot

    def open_lmdb(self):
        self.db = lmdb.open(self._path, map_size=1024 ** 4, readonly=True, lock=False)
        self.txn = self.db.begin()
        self.num = int(self.txn.get('num_samples'.encode()))
        self.datum = datum_pb2_genhead_new.Datum_new()

    def _load_raw_bg_feature(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        feature = np.fromstring(self.datum.bgs, dtype=np.float16)
    
        recon_feature = feature
        recon_feature = recon_feature.reshape(64,64,-1)
        recon_feature = recon_feature.transpose(2, 0, 1) # HWC => CHW

        return recon_feature

    def _load_raw_seg_render(self, raw_idx, frame_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        seg = np.fromstring(self.datum.segs_render, dtype=np.uint8).reshape(self.datum.num//5,5,-1)
    
        source_seg = seg[frame_idx[0],view_idx[0]]
        driving_seg = seg[frame_idx[1],view_idx[1]]
        recon_seg = seg[frame_idx[1],view_idx[2]]
        source_seg = source_seg.reshape(64,64,-1)
        driving_seg = driving_seg.reshape(64,64,-1)
        recon_seg = recon_seg.reshape(64,64,-1)
        if source_seg.ndim == 2:
            source_seg = source_seg[:, :, np.newaxis] # HW => HWC
        if driving_seg.ndim == 2:
            driving_seg = driving_seg[:, :, np.newaxis] # HW => HWC
        if recon_seg.ndim == 2:
            recon_seg = recon_seg[:, :, np.newaxis] # HW => HWC
        source_seg = source_seg.transpose(2, 0, 1) # HWC => CHW
        driving_seg = driving_seg.transpose(2, 0, 1) # HWC => CHW
        recon_seg = recon_seg.transpose(2, 0, 1) # HWC => CHW

        if source_seg.shape[0] == 1:
            source_seg = np.tile(source_seg, (3, 1, 1))

        if driving_seg.shape[0] == 1:
            driving_seg = np.tile(driving_seg, (3, 1, 1))

        if recon_seg.shape[0] == 1:
            recon_seg = np.tile(recon_seg, (3, 1, 1))

        return source_seg, driving_seg, recon_seg

    def __getitem__(self, idx):
        source_image, driving_image, recon_image, frame_idx, view_idx = self._load_raw_image(self._raw_idx[idx])
        source_seg, driving_seg, recon_seg = self._load_raw_seg(self._raw_idx[idx], frame_idx, view_idx)
        source_seg_render, driving_seg_render, recon_seg_render = self._load_raw_seg_render(self._raw_idx[idx], frame_idx, view_idx)
        source_depth, driving_depth, recon_depth = self._load_raw_depth(self._raw_idx[idx], frame_idx, view_idx)
        source_feature, driving_feature, recon_feature = self._load_raw_feature(self._raw_idx[idx], frame_idx, view_idx)
        recon_feature_bg = self._load_raw_bg_feature(self._raw_idx[idx])
        source_labels, driving_labels, recon_labels = self._load_raw_labels(self._raw_idx[idx], frame_idx, view_idx)
        source_motions, driving_motions, recon_motions = self._load_raw_motions(self._raw_idx[idx], frame_idx, view_idx)
        source_triplane, driving_triplane, recon_triplane = self._load_raw_triplane(self._raw_idx[idx], frame_idx)

        assert isinstance(source_image, np.ndarray)
        assert isinstance(driving_image, np.ndarray)
        assert isinstance(recon_image, np.ndarray)
        assert isinstance(source_seg, np.ndarray)
        assert isinstance(driving_seg, np.ndarray)
        assert isinstance(recon_seg, np.ndarray)
        assert isinstance(source_seg_render, np.ndarray)
        assert isinstance(driving_seg_render, np.ndarray)
        assert isinstance(source_seg_render, np.ndarray)
        assert list(source_image.shape) == self.image_shape
        assert source_seg.shape[1] == self.image_shape[1] and source_seg.shape[2] == self.image_shape[2]
        assert source_image.dtype == np.uint8
        
        # use flame parameters as motion embedding if use_flame_mot is true
        if self.use_flame_mot:
            source_motions = np.concatenate([self.get_exp_param_w_jaw_pose(source_labels),self.get_eye_pose_param(source_labels)],axis=0)
            driving_motions = np.concatenate([self.get_exp_param_w_jaw_pose(driving_labels),self.get_eye_pose_param(driving_labels)],axis=0)

        return source_image.copy(), driving_image.copy(), recon_image.copy(), recon_seg.copy(), recon_seg_render.copy(), recon_depth.copy(), recon_feature.copy(), recon_feature_bg.copy(), recon_triplane.copy(), self.get_label(recon_labels), self.get_shape_param(source_labels), self.get_exp_param(driving_labels), self.get_pose_param(driving_labels), self.get_eye_pose_param(driving_labels), source_motions.copy(), driving_motions.copy()

#--------------------------------------------------------------------------------------------------------
# Dataloader for GenHead generated static multiview data (include depth, feature maps, triplane features, background, and rendered segmentation)
class GenHeadReconSegLmdbFolderDatasetV2_StaticNew(GenHeadReconSegLmdbFolderDatasetV2_Static):

    def __init__(self,
        path,                   # Path to datalist.
        resolution      = None, # Ensure specific resolution, None = highest available.
        data_type      = "vfhq",# Deprecated 
        static         = False, # if True, only select multiview static frames instead of frames with different motions
        use_flame_mot  = False, # whether or not to use FLAME parameters as motion embedding
        **super_kwargs,         # Additional arguments for the Dataset base class.
        ):
        super().__init__(path=path, resolution=resolution, data_type=data_type, static=static, **super_kwargs)
        self.use_flame_mot = use_flame_mot

    def open_lmdb(self):
        self.db = lmdb.open(self._path, map_size=1024 ** 4, readonly=True, lock=False)
        self.txn = self.db.begin()
        self.num = int(self.txn.get('num_samples'.encode()))
        self.datum = datum_pb2_genhead_static_new.Datum_static_new()

    def _load_raw_bg_feature(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        feature = np.fromstring(self.datum.bgs, dtype=np.float16)
    
        recon_feature = feature
        recon_feature = recon_feature.reshape(64,64,-1)
        recon_feature = recon_feature.transpose(2, 0, 1) # HWC => CHW

        return recon_feature

    def _load_raw_seg_render(self, raw_idx, view_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        seg = np.fromstring(self.datum.segs_render, dtype=np.uint8).reshape(self.datum.num,-1)
    
        source_seg = seg[view_idx[0]]
        driving_seg = seg[view_idx[1]]
        recon_seg = seg[view_idx[2]]
        source_seg = source_seg.reshape(64,64,-1)
        driving_seg = driving_seg.reshape(64,64,-1)
        recon_seg = recon_seg.reshape(64,64,-1)
        if source_seg.ndim == 2:
            source_seg = source_seg[:, :, np.newaxis] # HW => HWC
        if driving_seg.ndim == 2:
            driving_seg = driving_seg[:, :, np.newaxis] # HW => HWC
        if recon_seg.ndim == 2:
            recon_seg = recon_seg[:, :, np.newaxis] # HW => HWC
        source_seg = source_seg.transpose(2, 0, 1) # HWC => CHW
        driving_seg = driving_seg.transpose(2, 0, 1) # HWC => CHW
        recon_seg = recon_seg.transpose(2, 0, 1) # HWC => CHW

        if source_seg.shape[0] == 1:
            source_seg = np.tile(source_seg, (3, 1, 1))

        if driving_seg.shape[0] == 1:
            driving_seg = np.tile(driving_seg, (3, 1, 1))

        if recon_seg.shape[0] == 1:
            recon_seg = np.tile(recon_seg, (3, 1, 1))

        return source_seg, driving_seg, recon_seg

    def __getitem__(self, idx):
        source_image, driving_image, recon_image, view_idx = self._load_raw_image(self._raw_idx[idx])
        source_seg, driving_seg, recon_seg = self._load_raw_seg(self._raw_idx[idx], view_idx)
        source_seg_render, driving_seg_render, recon_seg_render = self._load_raw_seg_render(self._raw_idx[idx], view_idx)
        source_depth, driving_depth, recon_depth = self._load_raw_depth(self._raw_idx[idx], view_idx)
        source_feature, driving_feature, recon_feature = self._load_raw_feature(self._raw_idx[idx], view_idx)
        recon_feature_bg = self._load_raw_bg_feature(self._raw_idx[idx])
        source_labels, driving_labels, recon_labels = self._load_raw_labels(self._raw_idx[idx], view_idx)
        source_motions, driving_motions, recon_motions = self._load_raw_motions(self._raw_idx[idx], view_idx)
        source_triplane, driving_triplane, recon_triplane = self._load_raw_triplane(self._raw_idx[idx])

        assert isinstance(source_image, np.ndarray)
        assert isinstance(driving_image, np.ndarray)
        assert isinstance(recon_image, np.ndarray)
        assert isinstance(source_seg, np.ndarray)
        assert isinstance(driving_seg, np.ndarray)
        assert isinstance(recon_seg, np.ndarray)
        assert isinstance(source_seg_render, np.ndarray)
        assert isinstance(driving_seg_render, np.ndarray)
        assert isinstance(source_seg_render, np.ndarray)
        assert list(source_image.shape) == self.image_shape
        assert source_seg.shape[1] == self.image_shape[1] and source_seg.shape[2] == self.image_shape[2]
        assert source_image.dtype == np.uint8
        
        # use flame parameters as motion embedding if use_flame_mot is true
        if self.use_flame_mot:
            source_motions = np.concatenate([self.get_exp_param_w_jaw_pose(source_labels),self.get_eye_pose_param(source_labels)],axis=0)
            driving_motions = np.concatenate([self.get_exp_param_w_jaw_pose(driving_labels),self.get_eye_pose_param(driving_labels)],axis=0)

        return source_image.copy(), driving_image.copy(), recon_image.copy(), recon_seg.copy(), recon_seg_render.copy(), recon_depth.copy(), recon_feature.copy(), recon_feature_bg.copy(), recon_triplane.copy(), self.get_label(recon_labels), self.get_shape_param(source_labels), self.get_exp_param(driving_labels), self.get_pose_param(driving_labels), self.get_eye_pose_param(driving_labels), source_motions.copy(), driving_motions.copy()