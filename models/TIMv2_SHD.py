from uu import encode

import torch
import torch.nn as nn
from requests import patch
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from braincog.model_zoo.base_module import BaseModule
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.strategy.surrogate import *
from functools import partial
from torchvision import transforms
from utils.MyNode import *
from models.TIM import *


class MLP(BaseModule):
    def __init__(self, in_features, step=10, encode_type='direct', hidden_features=None, out_features=None, drop=0.):
        super().__init__(step=step, encode_type=encode_type)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MyNode(step=step, tau=2.0)

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MyNode(step=step, tau=2.0)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()  # T B C N
        x = self.fc1_lif(x.flatten(0, 1)).reshape(T, B, self.c_hidden, H, W).contiguous()

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        x = self.fc2_lif(x.flatten(0, 1)).reshape(T, B, C, H, W).contiguous()
        return x

class TIMv2(BaseModule):
    '''
    :param: TIM_ratio should be integer
    '''
    def __init__(self, dim, step, encode_type='direct',TIM_ratio=4):
        super().__init__(step=step, encode_type=encode_type)
        self.step = step
        self.tim_ratio = TIM_ratio
        self.conv_up = nn.Conv3d(in_channels=self.step,out_channels=self.step*self.tim_ratio,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=False)
        self.bn1 = nn.BatchNorm3d(self.step*self.tim_ratio)
        self.lif1 = MyNode(step=step,tau=2.)

        self.conv_down = nn.Conv3d(in_channels=self.step*self.tim_ratio,out_channels=self.step,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=False,groups=self.step)
        self.bn2 = nn.BatchNorm3d(self.step)
        self.lif2 = MyNode(step=step,tau=2.)

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape

        x = x.transpose(0,1)
        x = self.bn1(self.conv_up(x)).transpose(0,1)
        x = self.lif1(x.flatten(0,1)).reshape(-1,B,C,H,W).contiguous()

        x = x.transpose(0,1)
        x = self.bn2(self.conv_down(x)).transpose(0,1)
        x = self.lif2(x.flatten(0,1)).reshape(T,B,C,H,W).contiguous()

        return x

class SSA(BaseModule):
    def __init__(self, dim, step=10, encode_type='direct', num_heads=8, TIM_ratio = 4, ):
        super().__init__(step=10, encode_type='direct')
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim

        self.num_heads = num_heads

        self.in_channels = dim // num_heads

        self.TIM = TIMv2(dim, step=step, TIM_ratio=TIM_ratio)
        # self.kTIM = TIMv2(dim, step=step, TIM_ratio=TIM_ratio)
        # self.vTIM = TIMv2(dim, step=step, TIM_ratio=TIM_ratio)

        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        # self.q_lif = MyNode(step=step, tau=2.0)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        # self.k_lif = MyNode(step=step, tau=2.0)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        # self.v_lif = MyNode(step=step, tau=2.0)

        self.res_lif = MyNode(step=step, tau=2.0)
        self.attn_lif = MyNode(step=step, tau=2.0, v_threshold=0.5, )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MyNode(step=step, tau=2.0, )


    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape
        N = H * W
        x_for_qkv = x.flatten(-2,-1).flatten(0, 1) # TB C N

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.TIM(q_conv_out).flatten(-2,-1).transpose(-2,-1)
        q = q_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.TIM(k_conv_out).flatten(-2, -1).transpose(-2, -1)
        k = k_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.TIM(v_conv_out).flatten(-2, -1).transpose(-2, -1)
        v = v_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # SSA
        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x.flatten(0, 1))
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, H, W)

        return x

class TWSA(BaseModule):
    def __init__(self, dim, step=10, encode_type='direct', num_heads=8,twsa_scale=0.25):
        super().__init__(step=step, encode_type=encode_type)
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim

        self.num_heads = num_heads

        self.in_channels = dim // num_heads

        self.scale = twsa_scale

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MyNode(step=step, tau=2.0)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MyNode(step=step, tau=2.0)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MyNode(step=step, tau=2.0)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MyNode(step=step, tau=2.0, )

        self.sdsa_lif = MyNode(step=step, tau=2.0)
        self.twsa_lif = MyNode(step=step, tau=2.0)

        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.attn_lif = MyNode(step=step,tau=2.0)
    def sdsa_cal(self, q, k, v, lif):
        T, B, H, N, CoH = q.shape  # CoH： C/H
        C = CoH * H

        kv = k.mul(v)  # point-wise multiplication
        kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        if lif is not None:
            kv = lif(kv.flatten(0, 1)).reshape(T, B, H, -1, CoH).contiguous()
        return q.mul(kv)

    def twsa_cal(self, q, k, v, lif):
        T, B, H, N, CoH = q.shape  # CoH： C/H

        q = q.permute(3,1,2,0,4)
        k = k.permute(3,1,2,0,4)
        v = v.permute(3,1,2,0,4)

        attn = (q @ k.transpose(-2, -1))

        r = (attn @ v) * self.scale
        r = r.permute(3,1,2,0,4)
        if lif is not None:
            r = lif(r).flatten(0,1).reshape(T, B, H, N, CoH).contiguous()
        return r

    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(-2,-1) # T B C N

        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        q = q_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        k = k_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        v = v_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # SDSA
        attn_sdsa = self.sdsa_cal(q, k, v, self.sdsa_lif)
        attn_sdsa = self.pool(attn_sdsa).transpose(3, 4).reshape(T, B, C, N).contiguous()
        # TWSA
        attn_twsa = self.sdsa_cal(q, k, v, self.twsa_lif).transpose(3, 4).reshape(T, B, C, N).contiguous()

        attn = (attn_twsa + attn_sdsa) * 0.5
        attn = self.attn_lif(attn.flatten(0, 1))

        x = self.proj_lif(self.proj_bn(self.proj_conv(attn))).reshape(T, B, C, H, W).contiguous()

        return x



class SPS_br1(BaseModule):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=1,
                 embed_dims=256,):
        super().__init__(step=step, encode_type=encode_type)
        self.image_size = [img_h, img_w]

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels


        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MyNode(step=step, tau=2.0)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MyNode(step=step, tau=2.0)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MyNode(step=step, tau=2.0)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MyNode(step=step, tau=2.0)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # May lose too much spatial features
        # Down sampling
        self.res_conv = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=1, stride=4, padding=0, bias=False)
        self.res_bn = nn.BatchNorm2d(embed_dims)
        self.res_lif = MyNode(step=step, tau=2.0)

    def forward(self, x):
        self.reset()

        T, B, C, N = x.shape

        x = F.interpolate(x, size=(32,32), mode='bilinear', align_corners=False)
        x = x.reshape(T, B, C, 32, 32).contiguous()

        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x.flatten(0, 1)).contiguous()  # No Maxpool here

        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.maxpool1(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif1(x.flatten(0, 1)).contiguous()

        x_feat = x
        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.maxpool2(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj_lif2(x.flatten(0, 1)).contiguous()

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.maxpool3(x.flatten(0,1)).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj_lif3(x.flatten(0, 1)).reshape(T, B, -1, H // 8, W // 8).contiguous()

        x_feat = self.res_conv(x_feat)
        x_feat = self.res_bn(x_feat).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x_feat = self.res_lif(x_feat.flatten(0, 1)).reshape(T, B, -1, H // 8, W // 8).contiguous()

        return x + x_feat  # T B Dim H//8 W//8


class SPS_br2(BaseModule):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=2,
                 embed_dims=256,):
        super().__init__(step=step, encode_type=encode_type)
        self.image_size = [img_h, img_w]

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels



        self.proj_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = MyNode(step=step, tau=2.0)


        self.proj_conv4 = nn.Conv2d(embed_dims , embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn4 = nn.BatchNorm2d(embed_dims)
        self.proj_lif4 = MyNode(step=step, tau=2.0)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # May lose too much spatial features
        # Down sampling
        # Down sampling rate of conv should equal to the rate of MP
        self.res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.res_bn = nn.BatchNorm2d(embed_dims)
        self.res_lif = MyNode(step=step, tau=2.0)

    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape
        x_feat = x

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x.flatten(0, 1)).contiguous()  # No Maxpool here

        x = self.proj_conv4(x)
        x = self.proj_bn4(x)
        x = self.maxpool4(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif4(x.flatten(0, 1)).reshape(T, B, -1, H // 2, W // 2).contiguous()

        x_feat = self.res_conv(x_feat.flatten(0, 1))
        x_feat = self.res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x_feat = self.res_lif(x_feat.flatten(0, 1)).reshape(T, B, -1, H // 2, W // 2).contiguous()

        return x + x_feat  # T B Dim H//8 W//8

class stage1_block(nn.Module):
    def __init__(self,dim, step=10, mlp_ratio=4.,num_heads=16,twsa_scale=0.25, ):
        super().__init__()

        self.attn = TWSA(dim,step=step, num_heads=num_heads,twsa_scale=twsa_scale)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(step=step, in_features=dim, hidden_features=mlp_hidden_dim,)

    def forward(self,x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x



class stage2_block(nn.Module):
    def __init__(self, dim, step=10, num_heads=16, mlp_ratio=4.,  drop=0., norm_layer=nn.LayerNorm, TIM_ratio=4):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = SSA(dim,step =step, num_heads=num_heads, TIM_ratio=TIM_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(step=step,in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x



class spikformer_TIMv2(nn.Module):
    def __init__(self, step=10, TIM_ratio=4, twsa_scale=0.25,
                 img_h=128, img_w=128, patch_size=16, in_channels=1, num_classes=10,
                 embed_dims=256, num_heads=16, mlp_ratios=4, drop_path_rate=0.,
                 depths=2, ):
        super().__init__()
        self.T = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed1 = SPS_br1(step=step,
                          img_h=img_h,
                          img_w=img_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims//2)


        stage1 = stage1_block(dim=embed_dims // 2, num_heads=num_heads,step=step,
                                             twsa_scale=twsa_scale, mlp_ratio=mlp_ratios)


        patch_embed2 =SPS_br2(step=step,
                          img_h=img_h,
                          img_w=img_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims,)

        stage2 = stage2_block(step=step,dim=embed_dims, num_heads=num_heads, TIM_ratio = TIM_ratio)


        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"stage1", stage1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"stage2", stage2)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")

        x = patch_embed1(x)
        x = stage1(x)

        x = patch_embed2(x)
        x = stage2(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.permute(1,0,2,3) # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

# Hyperparams could be adjust here

@register_model
def Spikformer_TIMv2_SHD(pretrained=False, **kwargs):
    model = spikformer_TIMv2(step=10,TIM_ratio=4,twsa_scale=0.25,embed_dims=256,img_h=64,img_w=64,
                             num_heads=8,num_classes=20,patch_size=16)
    model.default_cfg = _cfg()
    return model


