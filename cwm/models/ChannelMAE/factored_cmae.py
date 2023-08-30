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
        trunc_normal_(self.channel_pos_embed, std=self._channel_embed_std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'channel_pos_embed', 'cls_token'}

    def _init_pos_embed(self, use_learnable_pos_emb: bool = False) -> torch.Tensor:
        """The usual `pos_embed` that is returned now is tiled over the channel dimension."""

        num_patches = self.num_patches // self.num_channel_groups
        if use_learnable_pos_emb:
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        else:
            pos_embed = get_sinusoid_encoding_table(num_patches, self.embed_dim)

        return (
            pos_embed
            .unsqueeze(1)
            .repeat(1, self.num_channel_groups, 1, 1)
            .reshape(1, self.num_patches, self.embed_dim)
        )

    def _init_channel_pos_embed(self) -> torch.Tensor:
        """A learnable `channel_pos_embed` that varies only by channel idx"""
        
        num_patches_per_group = self.num_patches // self.num_channel_groups
        return (
            nn.Parameter(torch.zeros(1, self.num_channel_groups, self.embed_dim))
            .unsqueeze(2)
            .repeat(1, 1, num_patches_per_group, 1)
            .reshape(1, self.num_patches, self.embed_dim)
        )

    def tokenize(self, *args, **kwargs) -> torch.Tensor:

        # usual tokenization adds the spatial pos_embed
        x, mask = super(FactoredChannelMaeEncoder, self).tokenize(*args, **kwargs)

        # add the channel pos embed
        channel_pos_embed = self.channel_pos_embed.type_as(x).to(x.device).clone().detach()
        x = x + channel_pos_embed

        return (x, mask)

class FactoredChannelMae(ChannelMae):
    """ChannelMae that uses the FactoredChannelMaeEncoder and also has a factored decoder_pos_embed"""

    def __init__(self, *args, **kwargs) -> None:
        """Sets the decoder pos embed as a factored one, with learnable channel_pos_embed"""
        super(FactoredChannelMae, self).__init__(*args, **kwargs)
        self.pos_embed = self._set_factored_pos_embed()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'channel_pos_embed', 'cls_token', 'mask_token'}
    
    def _build_encoder(self, params: Dict = {}) -> nn.Module:
        return FactoredChannelMaeEncoder(**params)

    def _set_factored_pos_embed(self) -> None:
        """Recreate the positional embedding as a factored one, and initialize"""
        num_patches_per_group = self.encoder.num_patches // self.num_channel_groups
        pos_embed = get_sinusoid_encoding_table(num_patches_per_group, self.decoder.embed_dim)
        self.channel_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_channel_groups, self.decoder.embed_dim))
        trunc_normal_(self.channel_pos_embed, std=self.encoder._channel_embed_std)

        pos_embed = pos_embed.unsqueeze(1).repeat(1, self.num_channel_groups, 1, 1)
        chan_embed = self.channel_pos_embed.unsqueeze(2).repeat(1, 1, num_patches_per_group, 1)

        return (pos_embed + chan_embed).reshape(1, self.num_patches, self.decoder.embed_dim)
        

    
                     
            
    


