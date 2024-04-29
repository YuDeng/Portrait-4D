# PD-FGC motion encoder, modified from https://github.com/Dorniwang/PD-FGC-inference
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .FAN_feature_extractor import FAN_use

class FanEncoder(nn.Module):
    def __init__(self):
        super(FanEncoder, self).__init__()
        # pose_dim = self.opt.model.net_motion.pose_dim
        # eye_dim = self.opt.model.net_motion.eye_dim
        pose_dim = 6
        eye_dim = 6
        self.model = FAN_use()

        self.to_mouth = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim-eye_dim))
        
        self.to_headpose = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.headpose_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, pose_dim))

        self.to_eye = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.eye_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, eye_dim))

        self.to_emo = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.emo_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 30))

    def forward_feature(self, x):
        net = self.model(x)
        return net

    def forward(self, x):
        x = self.model(x)
        mouth_feat = self.to_mouth(x)
        headpose_feat = self.to_headpose(x)
        headpose_emb = self.headpose_embed(headpose_feat)
        eye_feat = self.to_eye(x)
        eye_embed = self.eye_embed(eye_feat)
        emo_feat = self.to_emo(x)
        emo_embed = self.emo_embed(emo_feat)
        return headpose_emb, eye_embed, emo_embed, mouth_feat
            