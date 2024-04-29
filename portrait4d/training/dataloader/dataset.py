# Dataloader for training GenHead, modified from EG3D: https://github.com/NVlabs/eg3d

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
from training.dataloader.protocols import datum_genhead_pb2 as datum_pb2
import lmdb
import cv2
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


class PortraitSynthesisSegLmdbFolderDatasetV2(Dataset):
    def __init__(self,
        path,                   # Path to datalist.
        resolution      = None, # Ensure specific resolution, None = highest available.
        data_type      = "vox2",# Set dataset type, deprecated
        rescale_camera = False, # Rescale camera extrinsics and intrinscs to align with an older version of camera labels
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        print(self._path)
        self._resolution = resolution
        self._zipfile = None
        self._data_type = data_type
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
        if resolution is None:
            self._resolution = raw_shape[2]
        name = os.path.splitext(os.path.basename(self._path))[0]
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def open_lmdb(self):
        self.db = lmdb.open(self._path, map_size=1024 ** 4, readonly=True, lock=False)
        self.txn = self.db.begin()
        self.num = int(self.txn.get('num_samples'.encode()))
        self.datum = datum_pb2.Datum_genhead()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._load_raw_labels(d.raw_idx).copy()
        return d

    def get_label_std(self):
        return 0

    @property
    def resolution(self):
        return self._resolution

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._load_raw_labels(0)
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
        return self._load_raw_labels(0).dtype == np.int64

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        image = np.fromstring(self.datum.image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]] # bgr -> rgb
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_seg(self, raw_idx):
        if self.txn is None:
            self.open_lmdb()
        value = self.txn.get('{:0>8d}'.format(raw_idx).encode())
        self.datum.ParseFromString(value)
        seg = np.fromstring(self.datum.seg, dtype=np.uint8)
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
        labels = np.fromstring(self.datum.labels, dtype=np.float32)
        intrinsics = labels[16:25].reshape(3,3)
        
        if self.rescale_camera:
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

    def get_label(self, idx):
        label = self._load_raw_labels(self._raw_idx[idx])[:25]
        return label.copy()

    def get_shape_param(self, idx):
        label = self._load_raw_labels(self._raw_idx[idx])[25:325]
        return label.copy()

    def get_exp_param(self, idx):
        label = self._load_raw_labels(self._raw_idx[idx])[325:425]
        return label.copy()

    def get_exp_param_w_jaw_pose(self, idx):
        label = self._load_raw_labels(self._raw_idx[idx])
        label = np.concatenate([label[325:425],label[428:431]],axis=0)
        return label.copy()

    def get_pose_param(self, idx):
        label = self._load_raw_labels(self._raw_idx[idx])[425:431]
        return label.copy()

    def get_eye_pose_param(self, idx):
        label = self._load_raw_labels(self._raw_idx[idx])[431:437]
        return label.copy()

    def get_label_all(self,idx):
        c = self.get_label(idx)
        shape_param = self.get_shape_param(idx)
        exp_param = self.get_exp_param(idx)
        pose_param = self.get_pose_param(idx)
        eye_pose_param = self.get_eye_pose_param(idx)
        return c, shape_param, exp_param, pose_param, eye_pose_param

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        seg = self._load_raw_seg(self._raw_idx[idx])

        assert isinstance(image, np.ndarray)
        assert isinstance(seg, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert seg.shape[1] == self.image_shape[1] and seg.shape[2] == self.image_shape[2]
        assert image.dtype == np.uint8

        return image.copy(), seg.copy(), self.get_label(idx), self.get_shape_param(idx), self.get_exp_param(idx), self.get_pose_param(idx), self.get_eye_pose_param(idx)