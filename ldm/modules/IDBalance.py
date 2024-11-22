import re
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
from einops import rearrange
from ldm.modules.attention import CrossAttention
import torch.nn.init as init

from ldm.modules.diffusionmodules.util import normalization


class IDBalance(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        
        ks = 3
        pw = ks // 2
        self.param_free_norm = normalization(norm_nc)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.norm = nn.LayerNorm(label_nc)
        self.crossattn = CrossAttention(query_dim=label_nc, context_dim=768,
                                    heads=8, dim_head=label_nc//8, dropout=0.)

        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        self.zero_init()
    def zero_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.zeros_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


    def forward(self, x_dic, segmap_dic, ref_cond):

        segmap = segmap_dic[str(x_dic.size(-1))]
        x = x_dic


        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        b, c, h, w = segmap.shape
        segmap = rearrange(segmap, 'b c h w -> b (h w) c')
        segmap = self.crossattn(self.norm(segmap), context=ref_cond) + segmap
        segmap = rearrange(segmap, 'b (h w) c -> b c h w', h=h, w=w)

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * gamma + beta

        return out

