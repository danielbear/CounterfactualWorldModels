"""ChannelMAEs that factor the positional embedding into spatial and channelwise components"""
import os
import math
import numpy as np

from typing import Tuple, List, Optional, Union, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from cwm.models.ChannelMAE.cmae import (
    ChannelMaeEncoder, SoftChannelMaeEncoder,
    ChannelMae, SoftChannelMae, SoftInputChannelMae
)

from cwm.models.VideoMAE.utils import (
    get_sinusoid_encoding_table,
    trunc_normal_
)

class FactoredChannelMaeEncoder(ChannelMaeEncoder):
    """
    Positional embedding is factored into two components:

    - pos_embed: encodes only the spatial dimensions within each channel group
    - channel_pos_embed: encodes only the channel group idx, and is learnable

    The spatial embedding `pos_embed` is added at the usual spot, right after tokenizing.
    The channelwise embedding `channel_pos_embed` is added after this.
    """
    def __init__(self, channel_embed_std: float = 1.0, *args, **kwargs):
        super(FactoredChannelMaeEncoder, self).__init__(*args, **kwargs)
        self.channel_pos_embed = self._init_channel_pos_embed()
        self._channel_embed_std = channel_embed_std
        trunc_normal_(self.channel_pos_embed, std=channel_embed_std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'channel_pos_embed', 'cls_token'}

    def _init_pos_embed(self, use_learnable_pos_embed: bool = False) -> torch.Tensor:
        """The usual `pos_embed` that is returned now is tiled over the channel dimension."""

        if use_learnable_pos_embed:
            pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens_per_channel_group, self.embed_dim))
        else:
            pos_embed = get_sinusoid_encoding_table(self.num_tokens_per_channel_group, self.embed_dim)

        return pos_embed

    def _init_channel_pos_embed(self) -> torch.Tensor:
        """A learnable `channel_pos_embed` that varies only by channel idx"""
        
        return nn.Parameter(torch.zeros(1, self.num_channel_groups, self.embed_dim), requires_grad=True)

    def _broadcast_pos_embed(self, pe: torch.Tensor) -> torch.Tensor:
        """Broadcasts either the `pos_embed` or the `channel_pos_embed`; which is inferred from shape"""
        N = pe.size(1)

        if N == self.num_tokens_per_channel_group:
            tile = lambda x: x.unsqueeze(1).repeat(1, self.num_channel_groups, 1, 1)
        elif N == self.num_channel_groups:
            tile = lambda x: x.unsqueeze(2).repeat(1, 1, self.num_tokens_per_channel_group, 1)
        else:
            raise ValueError("num_tokens must be num_channel_groups or num_tokens_per_channel_group")

        return tile(pe).reshape(pe.size(0), self.num_tokens, self.embed_dim)

    def _apply_pos_embed_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Overwrites usual tokenwise addition. Instead broadcasts `pos_embed` and `channel_pos_embed`"""

        B = x.size(0)

        # get spatial pos embed
        if not self._use_learnable_pos_embed:
            pos_embed = self.pos_embed.type_as(x).to(x.device).clone().detach()
        else:
            pos_embed = self.pos_embed.to(x)

        # broadcast across channel groups
        pos_embed = self._broadcast_pos_embed(pos_embed)

        # get channelwise pos embed and broacast spatially within each channel
        channel_pos_embed = self.channel_pos_embed.to(x)
        channel_pos_embed = self._broadcast_pos_embed(channel_pos_embed)

        x = x + pos_embed
        x = x + channel_pos_embed

        return x

class FactoredChannelMae(ChannelMae):
    """ChannelMae that uses the FactoredChannelMaeEncoder and also has a factored decoder_pos_embed"""

    def __init__(self, *args, **kwargs) -> None:
        """Sets the decoder pos embed as a factored one, with learnable channel_pos_embed"""
        super(FactoredChannelMae, self).__init__(*args, **kwargs)

        ## init the pos embeddings using the FactoredChannelMaeEncoder
        self.embed_dim = self.decoder.embed_dim
        self.pos_embed = FactoredChannelMaeEncoder._init_pos_embed(self, use_learnable_pos_embed=False)
        self.channel_pos_embed = FactoredChannelMaeEncoder._init_channel_pos_embed(self)
        trunc_normal_(self.channel_pos_embed, std=self.encoder._channel_embed_std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'channel_pos_embed', 'cls_token', 'mask_token'}
    
    def _build_encoder(self, params: Dict = {}) -> nn.Module:
        return FactoredChannelMaeEncoder(**params)

    def _get_expanded_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Add both the spatial and channelwise pos embed. Don't detach the latter!"""
        B = x.size(0)
        pos_embed = FactoredChannelMaeEncoder._broadcast_pos_embed(self, self.pos_embed)        
        pos_embed = pos_embed.expand(B, -1, -1).to(x).clone().detach()

        channel_pos_embed = FactoredChannelMaeEncoder._broadcast_pos_embed(self, self.channel_pos_embed)
        channel_pos_embed = channel_pos_embed.expand(B, -1, -1).to(pos_embed)

        return pos_embed + channel_pos_embed
        

    
                     
            
    


