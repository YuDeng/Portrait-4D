import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from util import util
from lib.models.networks.audio_network import ResNetSE, SEBasicBlock
from lib.models.networks.FAN_feature_extractor import FAN_use
from lib.models.networks.vision_network import ResNeXt50
from torchvision.models.vgg import vgg19_bn
from lib.models.networks.swin_transformer import SwinTransformer
from lib.models.networks.wavlm.wavlm import WavLM, WavLMConfig


class ResSEAudioEncoder(nn.Module):
    def __init__(self, opt, nOut=2048, n_mel_T=None):
        super(ResSEAudioEncoder, self).__init__()
        self.nOut = nOut
        self.opt = opt
        pose_dim = self.opt.model.net_nonidentity.pose_dim
        eye_dim = self.opt.model.net_nonidentity.eye_dim
        # Number of filters
        num_filters = [32, 64, 128, 256]
        if n_mel_T is None: # use it when use audio identity
            n_mel_T = opt.model.net_audio.n_mel_T
        self.model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, self.nOut, n_mel_T=n_mel_T)
        if opt.audio_only:
            self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim))
        else:
            self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim-eye_dim))
            

    # def forward_feature(self, x):
    def forward(self, x, _type=None):
        input_size = x.size()
        if len(input_size) == 5:
            bz, clip_len, c, f, t = input_size
            x = x.view(bz * clip_len, c, f, t)
        out = self.model(x)
        
        if _type == "to_mouth_embed":

            out = out.view(-1, out.shape[-1])
            mouth_embed = self.mouth_embed(out)
            return out, mouth_embed

        return out

    # def forward(self, x):
    #     out = self.forward_feature(x)
    #     score = self.fc(out)
    #     return out, score


class ResSESyncEncoder(ResSEAudioEncoder):
    def __init__(self, opt):
        super(ResSESyncEncoder, self).__init__(opt, nOut=512, n_mel_T=1)


class ResNeXtEncoder(ResNeXt50):
    def __init__(self, opt):
        super(ResNeXtEncoder, self).__init__(opt)


class VGGEncoder(nn.Module):
    def __init__(self, opt):
        super(VGGEncoder, self).__init__()
        self.model = vgg19_bn(num_classes=opt.data.num_classes)

    def forward(self, x):
        return self.model(x)


class FanEncoder(nn.Module):
    def __init__(self, opt):
        super(FanEncoder, self).__init__()
        self.opt = opt
        pose_dim = self.opt.model.net_nonidentity.pose_dim
        eye_dim = self.opt.model.net_nonidentity.eye_dim
        self.model = FAN_use()
        # self.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, opt.data.num_classes))

        # mapper to mouth subspace

        ### revised version1 
        # self.to_mouth = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        # self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim-eye_dim))
        
        # mapper to head pose subspace

        ### revised version1
        self.to_headpose = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.headpose_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, pose_dim))

        self.to_eye = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.eye_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, eye_dim))

        self.to_emo = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.emo_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 30))
        # self.feature_fuse = nn.Sequential(nn.ReLU(), nn.Linear(1036, 512))
        # self.feature_fuse = nn.Sequential(nn.ReLU(), nn.Linear(1036, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))

    def forward_feature(self, x):
        net = self.model(x)
        return net

    def forward(self, x, _type="feature"):
        if _type == "feature":
            return self.forward_feature(x)
        elif _type == "feature_embed":
            x = self.model(x)
            # mouth_feat = self.to_mouth(x)
            # mouth_emb = self.mouth_embed(mouth_feat)
            headpose_feat = self.to_headpose(x)
            headpose_emb = self.headpose_embed(headpose_feat)
            eye_feat = self.to_eye(x)
            eye_embed = self.eye_embed(eye_feat)
            emo_feat = self.to_emo(x)
            emo_embed = self.emo_embed(emo_feat)
            # return headpose_emb, eye_embed, emo_feat
            return headpose_emb, eye_embed, emo_embed
        elif _type == "to_headpose":
            x = self.model(x)
            headpose_feat = self.to_headpose(x)
            headpose_emb = self.headpose_embed(headpose_feat)
            return headpose_emb
            

# class WavlmEncoder(nn.Module):

#     def __init__(self, opt):
#         super(WavlmEncoder, self).__init__()

#         wavlm_checkpoint = torch.load(opt.model.net_audio.official_pretrain)
#         wavlm_cfg = WavLMConfig(wavlm_checkpoint['cfg'])
#         # pose_dim = opt.model.net_nonidentity.pose_dim
        
#         self.model = WavLM(wavlm_cfg)
#         # self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(768, 512-pose_dim))

#     def forward(self, x):
#         feature = self.model.extract_features(x)[0]
#         # audio_feat = self.mouth_embed(feature.mean(1))

#         return feature.mean(1)


class WavlmEncoder(nn.Module):

    def __init__(self, opt):
        super(WavlmEncoder, self).__init__()

        self.input_wins = opt.audio.num_frames_per_clip
        self.s = (self.input_wins - 5) // 2 * 2
        self.e = self.s + 5 * 2 - 1

        wavlm_checkpoint = torch.load(opt.model.net_audio.official_pretrain)
        wavlm_cfg = WavLMConfig(wavlm_checkpoint['cfg'])
        pose_dim = opt.model.net_nonidentity.pose_dim
        
        self.model = WavLM(wavlm_cfg)
        self.mouth_feat = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 512))
        self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim))

    def forward(self, x):
        feature = self.model.extract_features(x)[0]
        feature = self.mouth_feat(feature[:, self.s:self.e].mean(1))
        audio_feat = self.mouth_embed(feature)

        return feature, audio_feat

class SwinEncoder(nn.Module):
    def __init__(self, cfg):
        super(SwinEncoder, self).__init__()

        self.encoder = SwinTransformer(
            num_classes = 0,
            img_size = cfg.model.net_nonidentity.img_size,
            patch_size = cfg.model.net_nonidentity.patch_size,
            in_chans = cfg.model.net_nonidentity.in_chans,
            embed_dim = cfg.model.net_nonidentity.embed_dim,
            depths = cfg.model.net_nonidentity.depths,
            num_heads = cfg.model.net_nonidentity.num_heads,
            window_size = cfg.model.net_nonidentity.window_size,
            mlp_ratio = cfg.model.net_nonidentity.mlp_ratio,
            qkv_bias = cfg.model.net_nonidentity.qkv_bias,
            qk_scale = None if not cfg.model.net_nonidentity.qk_scale else 0.1,
            drop_rate = cfg.model.net_nonidentity.drop_rate,
            drop_path_rate = cfg.model.net_nonidentity.drop_path_rate,
            ape = cfg.model.net_nonidentity.ape,
            patch_norm = cfg.model.net_nonidentity.patch_norm,
            use_checkpoint = False
        )

        # self.audio_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.eye_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.landmark_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.exp_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))

    def forward(self, img):

        feature = self.encoder(img)

        # audio_embed = self.audio_mlp(feature)
        # eye_embed = self.eye_mlp(feature)
        # ldmk_embed = self.landmark_mlp(feature)
        # exp_embed = self.exp_mlp(feature)

        # return feature, audio_embed, eye_embed, ldmk_embed, exp_embed
        return feature


class ResEncoder(nn.Module):
    def __init__(self, opt):
        super(ResEncoder, self).__init__()
        self.opt = opt
        self.model = resnet50(num_classes=512, include_top=True)

    def forward(self, x):
        feature = self.model(x)
        # print(feature.shape)
        return feature

class FansEncoder(nn.Module):
    def __init__(self, cfg):
        super(FansEncoder, self).__init__()

        self.encoder = FAN_use(out_dim=768)

        # self.audio_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.eye_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.landmark_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.exp_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))

    def forward(self, img):

        feature = self.encoder(img)

        # audio_embed = self.audio_mlp(feature)
        # eye_embed = self.eye_mlp(feature)
        # ldmk_embed = self.landmark_mlp(feature)
        # exp_embed = self.exp_mlp(feature)

        # return feature, audio_embed, eye_embed, ldmk_embed, exp_embed
        return feature
        
class SwinEncoder(nn.Module):
    def __init__(self, cfg):
        super(SwinEncoder, self).__init__()

        self.encoder = SwinTransformer(
            num_classes = 0,
            img_size = cfg.model.net_nonidentity.img_size,
            patch_size = cfg.model.net_nonidentity.patch_size,
            in_chans = cfg.model.net_nonidentity.in_chans,
            embed_dim = cfg.model.net_nonidentity.embed_dim,
            depths = cfg.model.net_nonidentity.depths,
            num_heads = cfg.model.net_nonidentity.num_heads,
            window_size = cfg.model.net_nonidentity.window_size,
            mlp_ratio = cfg.model.net_nonidentity.mlp_ratio,
            qkv_bias = cfg.model.net_nonidentity.qkv_bias,
            qk_scale = None if not cfg.model.net_nonidentity.qk_scale else 0.1,
            drop_rate = cfg.model.net_nonidentity.drop_rate,
            drop_path_rate = cfg.model.net_nonidentity.drop_path_rate,
            ape = cfg.model.net_nonidentity.ape,
            patch_norm = cfg.model.net_nonidentity.patch_norm,
            use_checkpoint = False
        )

        # self.audio_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.eye_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.landmark_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.exp_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))

    def forward(self, img):

        feature = self.encoder(img)

        # audio_embed = self.audio_mlp(feature)
        # eye_embed = self.eye_mlp(feature)
        # ldmk_embed = self.landmark_mlp(feature)
        # exp_embed = self.exp_mlp(feature)

        # return feature, audio_embed, eye_embed, ldmk_embed, exp_embed
        return feature


class ResEncoder(nn.Module):
    def __init__(self, opt):
        super(ResEncoder, self).__init__()
        self.opt = opt
        self.model = resnet50(num_classes=512, include_top=True)

    def forward(self, x):
        feature = self.model(x)
        # print(feature.shape)
        return feature

class FansEncoder(nn.Module):
    def __init__(self, cfg):
        super(FansEncoder, self).__init__()

        self.encoder = FAN_use(out_dim=768)

        # self.audio_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.eye_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.landmark_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))
        # self.exp_mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 768))

    def forward(self, img):

        feature = self.encoder(img)

        # audio_embed = self.audio_mlp(feature)
        # eye_embed = self.eye_mlp(feature)
        # ldmk_embed = self.landmark_mlp(feature)
        # exp_embed = self.exp_mlp(feature)

        # return feature, audio_embed, eye_embed, ldmk_embed, exp_embed
        return feature