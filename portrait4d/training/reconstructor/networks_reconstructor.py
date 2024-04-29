# Network components for Portrait4D's reconstructor
import os
import sys
import numpy as np
from typing import Optional
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch_utils import persistence

from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.encoders.resnet import resnet_encoders
from models.deeplabv3.decoder import DeepLabV3Decoder
from models.unet.openaimodel import UNetModel, Upsample, Downsample
from models.mix_transformer.mix_transformer import OverlapPatchEmbed, Block, BlockCross
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


encoders = {}
encoders.update(resnet_encoders)

def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, norm_layer=None, **kwargs):

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    params.update(norm_layer=norm_layer)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights,
                    name,
                    list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        encoder.load_state_dict(model_zoo.load_url(settings["url"]), strict=False)

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder

def AddCoord(im):
    B, C, H, W = im.shape

    y, x = torch.meshgrid(torch.linspace(-1, 1, H, dtype=torch.float32, device=im.device), torch.linspace(-1, 1, W, dtype=torch.float32, device=im.device), indexing='ij')
    xy = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(B,1,1,1) #(B,H,W,2)
    xy = xy.permute(0,3,1,2) #(B,2,H,W)

    ret = torch.cat([im, xy], dim=1)
    
    return ret

# Global appearance encoder, remove all bn layers, change input dimension to 5, and remove segmentation head following Live3DPortrait: https://arxiv.org/abs/2305.02310
@persistence.persistent_class
class EncoderGlobal(SegmentationModel): 
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: int = 256,
        in_channels: int = 5,
        activation: Optional[str] = None,
        aux_params: Optional[dict] = None,
        norm_layer: nn.Module = nn.Identity
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=8,
            norm_layer=norm_layer
        )

        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=decoder_channels,
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        x = AddCoord(x)
        # self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        return decoder_output

# Detail appearance encoder
@persistence.persistent_class
class EncoderDetail(nn.Module):
    def __init__(
        self,
        in_channels: int = 5
        ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def forward(self, x):
        
        x = AddCoord(x)
        output = self.encoder(x)

        return output

# Canonicalization and reenactment module
@persistence.persistent_class
class EncoderCanonical(nn.Module):
    def __init__(self, img_size=64, patch_size=3, in_chans=256, embed_dims=1024, mot_dims=512, mot_dims_hidden=512,
                 H_y=8, W_y=8, num_heads=4, mlp_ratios=2, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., num_blocks_neutral=3, num_blocks_motion=3, norm_layer=nn.LayerNorm,
                 sr_ratios=1, mapping_layers=0):
        super().__init__()
        self.num_blocks_neutral = num_blocks_neutral
        self.num_blocks_motion = num_blocks_motion
        self.mapping_layers = mapping_layers
        self.H_y = H_y
        self.W_y = W_y
        

        # mapping net for motion feature
        if mapping_layers > 0:
            self.maps = nn.ModuleList([])
            for i in range(mapping_layers):
                in_dims = mot_dims if i == 0 else mot_dims_hidden
                self.maps.append(nn.Linear(in_dims, mot_dims_hidden, bias=True))
                self.maps.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

            self.maps_neutral = nn.ModuleList([])
            for i in range(mapping_layers):
                in_dims = mot_dims if i == 0 else mot_dims_hidden
                self.maps_neutral.append(nn.Linear(in_dims, mot_dims_hidden, bias=True))
                self.maps_neutral.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))            
        else:
            self.maps = None
            self.maps_neutral = None
            mot_dims_hidden = mot_dims
        
        self.proj_y_neutral = nn.Linear(mot_dims_hidden, H_y * W_y * embed_dims, bias=True)
        self.proj_y = nn.Linear(mot_dims_hidden, H_y * W_y * embed_dims, bias=True)


        # patch_embed
        self.patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size, stride=2, in_chans=in_chans,
                                              embed_dim=embed_dims)

        # canonicalization blocks
        self.trans_blocks_neutral = nn.ModuleList([BlockCross(
            dim=embed_dims, dim_y=mot_dims_hidden, H_y=H_y, W_y=W_y, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
            sr_ratio=sr_ratios)
            for i in range(num_blocks_neutral)])
        
        # reenactment blocks
        self.trans_blocks_motion = nn.ModuleList([BlockCross(
            dim=embed_dims, dim_y=mot_dims_hidden, H_y=H_y, W_y=W_y, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
            sr_ratio=sr_ratios)
            for i in range(num_blocks_motion)])
    
        
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        
        self.convs = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_chans, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, np.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}  # has pos_embed may be better


    def forward_features(self, x, y, y_x, scale=1.):
        B = x.shape[0]
        outs = []

        if self.maps is not None:
            for layer in self.maps:
                y = layer(y)

        if self.maps_neutral is not None:
            for layer in self.maps_neutral:
                y_x = layer(y_x)    
        
        
        y = self.proj_y(y).reshape(B, self.H_y*self.W_y, -1)
        y_x = self.proj_y_neutral(y_x).reshape(B, self.H_y*self.W_y, -1)
        

        # trans blocks
        x, H, W = self.patch_embed(x)

        # neutralize the face
        for i, blk in enumerate(self.trans_blocks_neutral):
            x = blk(x, y_x, H, W, scale=scale)
        
        # animate the face
        for i, blk in enumerate(self.trans_blocks_motion):
            x = blk(x, y, H, W, scale=scale)        
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.pixelshuffle(x)

        outs.append(x)

        # conv blocks
        x = self.convs(x)
        outs.append(x)

        return outs

    def forward(self, x, y, y_x, scale=1.):
        x = self.forward_features(x, y, y_x, scale=scale)

        return x

# Triplane decoder
@persistence.persistent_class
class DecoderTriplane(nn.Module):
    def __init__(self, img_size=256, patch_size=3, embed_dims=1024,
                 num_heads=2, mlp_ratios=2, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., num_blocks=1, norm_layer=nn.LayerNorm,
                 sr_ratios=2):
        super().__init__()
        self.num_blocks = num_blocks

        self.convs1 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # patch_embed
        self.patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size, stride=2, in_chans=128, embed_dim=embed_dims)

        # transformer encoder
        self.trans_blocks = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
            sr_ratio=sr_ratios)
            for i in range(num_blocks)])
        
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        
        self.convs2 = nn.Sequential(
            nn.Conv2d(352, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)
        )

        self.conv_last = nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)
        self.conv_last.apply(self._init_weights_last)

    def _init_weights_last(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.15)
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, np.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}  # has pos_embed may be better


    def forward(self, x_global, x_detail):

        x = torch.cat([x_global, x_detail], dim=1) # [B,C,H,W]
        B = x.shape[0]

        # convs1
        x = self.convs1(x)

        # trans blocks
        x, H, W = self.patch_embed(x)
        for i, blk in enumerate(self.trans_blocks):
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.pixelshuffle(x)

        x = torch.cat([x,x_global], dim=1)

        # convs2
        x = self.convs2(x)

        x = self.conv_last(x)

        return x

@persistence.persistent_class
class EncoderBG(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=32,
        ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        ) # 128

        self.down_conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            Downsample(64, True, dims=2, out_channels=128)
        ) # 64

        self.down_conv_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            Downsample(128, True, dims=2, out_channels=256)
        ) # 32

        self.down_conv_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            Downsample(256, True, dims=2, out_channels=512)
        ) # 16

        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.up_conv_1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            Upsample(256, True, dims=2, out_channels=256)
        ) # 32

        self.up_conv_2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            Upsample(128, True, dims=2, out_channels=128)
        ) # 64

        self.up_conv_3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            Upsample(64, True, dims=2, out_channels=64)
        ) # 128

        self.up_conv_4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            Upsample(64, True, dims=2, out_channels=64)
        ) # 256

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, np.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        
        x = AddCoord(x)

        h1 = self.input_conv(x)
        h2 = self.down_conv_1(h1)
        h3 = self.down_conv_2(h2)
        h4 = self.down_conv_3(h3)

        h = self.middle(h4)

        x = self.up_conv_1(torch.cat([h, h4], 1))
        x = self.up_conv_2(torch.cat([x, h3], 1))
        x = self.up_conv_3(torch.cat([x, h2], 1))
        x = self.up_conv_4(torch.cat([x, h1], 1))
        output = self.out_conv(x)

        return output