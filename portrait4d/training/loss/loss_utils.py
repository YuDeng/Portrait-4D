# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

# Distortion loss in MipNeRF-360: https://github.com/google-research/multinerf
def lossfun_distortion(t, w, reduction='mean'):
  """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
  # The loss incurred between all pairs of intervals.
  # print("t:",t.shape)
  # print("w:",w.shape)
  ut = (t[..., 1:] + t[..., :-1]) / 2
  dut = torch.abs(ut[..., :, None] - ut[..., None, :])
  loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

  # The loss incurred within each individual interval with itself.
  loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

  if reduction is None:
    return loss_inter + loss_intra
  return (loss_inter + loss_intra).mean()