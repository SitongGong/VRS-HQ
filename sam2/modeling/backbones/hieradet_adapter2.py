# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        # add_adapter: bool = False,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        # 是否在该层引入多尺度特征
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )
        # assert add_adapter == False or q_stride is not None

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # num_conv, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))     # x: bs, h, w, dim
        
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """
    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
        add_adapter: bool = True,
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride    # 下采样步长
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]       # 各个阶段结束时的block索引     1, 7, 43, 47
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]      # 2, 8, 44
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()
        
        self.adapter_block = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                
                # 如果进行pooling，则引入adapter层
                self.add_adapter = add_adapter
                if self.add_adapter:
                    self.adapter_block.append(AdapterBlock1(dim=embed_dim, visual_in_dim=embed_dim, seg_token_dim=256, num_heads=num_heads, dropout=0.0))
                
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                    
            if i == depth - 1:
                self.adapter_block.append(AdapterBlock1(dim=dim_out, visual_in_dim=dim_out, seg_token_dim=256, num_heads=num_heads, dropout=0.0))

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                # add_adapter=True if i in self.q_pool_blocks else False,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor, seg_embedding=None) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): num_conv, h, w, 3
            seg_embedding (_type_): num_conv, 1, dim=256

        Returns:
            List[torch.Tensor]: _description_
        """
        
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        cont = 0
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                H, W, x_dim  = x.shape[1:]
                # 如果seg token存在时，在两个阶段之间融入seg token中的特征
                if self.add_adapter and seg_embedding is not None:   # 对于后几帧的图像特征，不引入seg token
                    x = x.flatten(1, 2)      # bs, h * w, dim
                    x = self.adapter_block[cont](x, seg_embedding)
                    x = x.view(-1, H, W, x_dim)
                cont += 1
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs      # list: num_conv, dim, h, w
    
    
################ 方案二为严格的adapter，在每个block中均需要加入 ##################
    
class CrossModalAttention2(nn.Module):
    def __init__(self, num_prompt_tokens, visual_dim, seg_token_dim, out_dim, key_dim, value_dim, num_heads):
        super(CrossModalAttention2, self).__init__()
        
        self.visual_dim = visual_dim
        self.seg_token_dim = seg_token_dim
        self.out_dim = out_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        
        self.prompt_query = nn.Parameter()
        self.self_attn = nn.MultiheadAttention(self.visual_dim, self.num_heads, kdim=self.key_dim, vdim=self.value_dim)
        
        # keys: seg token features: bs, num_conv, seg_token_dim
        # conv1d or nn.linear
        self.f_query = nn.Sequential(nn.Conv1d(self.visual_dim, self.key_dim, kernel_size=1, stride=1),
                                     nn.InstanceNorm1d(self.key_dim))
        
        # self.f_key = nn.Sequential(nn.Conv1d(self.seg_token_dim, self.key_dim, kernel_size=1, stride=1))
        self.f_key = nn.Linear(self.seg_token_dim, self.key_dim)
        # self.f_value = nn.Sequential(nn.Conv1d(self.seg_token_dim, self.value_dim, kernel_size=1, stride=1))
        self.f_value = nn.Linear(self.seg_token_dim, self.value_dim)
        
        self.f_out = nn.Sequential(nn.Conv1d(self.value_dim, self.out_dim, kernel_size=1, stride=1),
                                   nn.InstanceNorm1d(self.out_dim))

    def forward(self, visual_feat: torch.Tensor, seg_token: torch.Tensor):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        
        B, HW = visual_feat.size(0), visual_feat.size(1)
        visual_feat = visual_feat.permute(0, 2, 1)  # (B, key_channels, H*W)

        query = self.f_query(visual_feat)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        
        key = self.f_key(seg_token).permute(0, 2, 1)  # (B, key_channels, 1)
        value = self.f_value(seg_token).permute(0, 2, 1)  # (B, self.value_channels, 1)
        n_l = value.size(-1)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        query = query.reshape(B, HW, self.num_heads, self.key_dim // self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, self.key_channels//self.num_heads, 1)
        key = key.reshape(B, self.num_heads, self.key_dim//self.num_heads, n_l)
        # (b, num_heads, self.value_channels//self.num_heads, 1)
        value = value.reshape(B, self.num_heads, self.value_dim//self.num_heads, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, 1)
        sim_map = (self.key_dim ** (-0.5)) * sim_map  # scaled dot product
        sim_map = F.softmax(sim_map, dim=-2)  # (B, num_heads, H * W, 1)
        
        out = sim_map * value  # (B, num_heads, H * W, value_dim // num_heads)
        
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_dim)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.f_out(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out


class AdapterBlock2(nn.Module):
    def __init__(self, dim, visual_in_dim, seg_token_dim, num_heads, dropout):
        super(AdapterBlock2, self).__init__()
        
        self.vis_lang_attn = CrossModalAttention(visual_in_dim, seg_token_dim, key_dim=dim, value_dim=dim, out_dim=dim, num_heads=num_heads)
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        
    def forward(self, visual_feat, seg_token):
        # visual_feat: bs, h * w, dim
        # seg_token: bs, 1, seg_token_dim
        # visual_feat_residual = self.vis_project(visual_feat.permute(0, 2, 1))
        
        visual_feat_ = self.vis_lang_attn(visual_feat, seg_token)
        
        # visual_feat_ = visual_feat_.permute(0, 2, 1)
        # visual_feat_ = torch.mul(visual_feat_residual, visual_feat_)     # bs, dim, h * w
        # visual_feat_ = self.out_project(visual_feat_)      # bs, dim, h * w
        
        # visual_feat_ = visual_feat_.permute(0, 2, 1)      # bs, h * w, dim
        visual_feat = self.res_gate(visual_feat_) * visual_feat_ + visual_feat
        
        return visual_feat     # bs, h * w, dim
