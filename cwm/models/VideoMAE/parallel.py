from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cwm.models.VideoMAE.utils import Attention, DropPath, Mlp

_LayerNormNoBias = partial(nn.LayerNorm, eps=1e-6, elementwise_affine=False)

class ParallelScalingBlock(nn.Module):
    """Based on `Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442"""
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_norm=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            init_values=None,
            act_layer=nn.GELU,
            norm_layer=_LayerNormNoBias,
            in_dim=None,
            flash_attention=False
    ) -> None:
        
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.in_dim = in_dim or self.dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        mlp_hidden_dim = int(dim * mlp_ratio)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        # initial norm and linear proj on inputs
        self.in_norm = norm_layer(self.in_dim)
        self.in_proj = nn.Linear(self.in_dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + 3 * [self.dim]

        # norms and biases
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.flash_attention = flash_attention
        if self.flash_attention:
            from flash_attn import flash_attn_qkvpacked_func
            self.fa = lambda qkv: flash_attn_qkvpacked_func(
                qkv, dropout_p=self.attn_drop_rate, softmax_scale=1)        

        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(self.dim, self.dim, bias=True)

        self.mlp_drop = nn.Dropout(drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, self.dim, bias=True)

        if (init_values or 0) > 0:
            self.gamma = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, N, C = x.shape

        
    
        


