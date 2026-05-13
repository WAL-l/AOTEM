#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/23 13:22
# @Author  : Ws
# @File    : model.py
# @Software: PyCharm
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp
from monai.utils import optional_import
import torch.nn.functional as F
import numpy as np
rearrange, _ = optional_import("einops", name="rearrange")

@torch.no_grad()
def generate_octree_masks(V, s, D_m, H_m, W_m):
    """
    Efficient generation of binary octree masks using max pooling for 3D volumes.

    Parameters:
    - V (torch.Tensor): A batch of low-quality 3D volumes of shape (B, C, D, H, W).
    - s (float): Threshold value.
    - D_m (int): Desired mask depth.
    - H_m (int): Desired mask height.
    - W_m (int): Desired mask width.

    Returns:
    - M (torch.Tensor): Binary octree masks of shape (B, D_m, H_m, W_m) with values {0, 1}.
    """

    # Step 1: Get volume dimensions
    _, _, D, H, W = V.shape
    
    # Step 2: Determine the scale level `l`
    l_d = max(int(np.log2(D / D_m)), 0) if D_m > 0 else 0
    l_h = max(int(np.log2(H / H_m)), 0) if H_m > 0 else 0
    l_w = max(int(np.log2(W / W_m)), 0) if W_m > 0 else 0
    l = max(l_d, l_h, l_w, 1)  # Ensure l is at least 1

    # Step 3: Max pooling on V with kernel size 2^l x 2^l x 2^l to obtain V^+
    kernel_size = 2 ** l
    V_plus = F.max_pool3d(V, kernel_size=kernel_size)

    # Step 4: Max pooling on -V with kernel size 2^l x 2^l x 2^l to obtain V^-
    V_minus = F.max_pool3d(-V, kernel_size=kernel_size)

    # Step 5: Generate mask M based on threshold `s`
    # Compute the range (max - min) across channels and spatial dimensions
    M = ((V_plus + V_minus).mean(dim=1) >= s).float()

    # Step 6: Resize M to desired mask dimensions (D_m, H_m, W_m) using nearest-neighbor interpolation
    M = F.interpolate(M.unsqueeze(1), size=(D_m, H_m, W_m), mode='nearest').squeeze(1)

    # Step 7: Return the binary mask M
    return M


class AdaptiveBlockEmbedding(nn.Module):
    def __init__(self, in_channels=1, hidden_size=384, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # Patch embedding: 将每个patch投影到hidden_size维度
        self.patch_embed = nn.Conv3d(
            in_channels, 
            hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 用于合并块的embedding（处理2x2x2或更大的合并块）
        self.merged_patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, hidden_size // 2, kernel_size=patch_size * 2, stride=patch_size * 2),
            nn.GELU(),
            nn.Conv3d(hidden_size // 2, hidden_size, kernel_size=1)
        )
        
        # 位置编码
        self.pos_embed = None
        
    def compute_num_patches(self, D, H, W):
        return D // self.patch_size, H // self.patch_size, W // self.patch_size
    
    def get_positional_encoding(self, num_patches, device):
        d_patches, h_patches, w_patches = num_patches
        total_patches = d_patches * h_patches * w_patches
        
        if self.pos_embed is None or self.pos_embed.shape[1] != total_patches:
            # 使用正弦位置编码
            pos = torch.arange(total_patches, device=device).unsqueeze(1)
            dim = torch.arange(self.hidden_size, device=device).unsqueeze(0)
            
            div_term = torch.exp(dim.float() * (-np.log(10000.0) / self.hidden_size))
            pos_encoding = torch.sin(pos * div_term)
            
            self.pos_embed = pos_encoding.unsqueeze(0)  # [1, total_patches, hidden_size]
        
        return self.pos_embed
    
    def extract_patches(self, x):
        B, C, D, H, W = x.shape
        p = self.patch_size
        
        # 计算每个维度的patch数量
        d_patches = D // p
        h_patches = H // p
        w_patches = W // p
        
        # 重塑为patches
        # [B, C, D, H, W] -> [B, C, d_patches, p, h_patches, p, w_patches, p]
        x = x.view(B, C, d_patches, p, h_patches, p, w_patches, p)
        # [B, C, d_patches, h_patches, w_patches, p, p, p]
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        # [B, num_patches, C, p, p, p]
        x = x.view(B, d_patches * h_patches * w_patches, C, p, p, p)
        
        return x
    
    def forward(self, x, mask):
        B, C, D, H, W = x.shape
        _, D_m, H_m, W_m = mask.shape

        d_patches = D // self.patch_size
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches_total = d_patches * h_patches * w_patches

        mask_resized = F.interpolate(
            mask.unsqueeze(1), 
            size=(d_patches, h_patches, w_patches), 
            mode='nearest'
        ).squeeze(1)  # [B, d_patches, h_patches, w_patches]

        patches = self.extract_patches(x)  # [B, num_patches_total, C, p, p, p]

        mask_flat = mask_resized.reshape(B, -1)  # [B, num_patches_total]

        tokens_list = []
        
        for b in range(B):
            batch_mask = mask_flat[b]  # [num_patches_total]
            change_indices = torch.where(batch_mask == 1)[0]
            smooth_indices = torch.where(batch_mask == 0)[0]
            
            batch_tokens = []

            if len(change_indices) > 0:
                change_patches = patches[b, change_indices]  # [N_change, C, p, p, p]

                change_tokens = self.patch_embed(change_patches)  # [N_change, hidden_size, 1, 1, 1]
                change_tokens = change_tokens.view(len(change_indices), self.hidden_size)
                batch_tokens.append(change_tokens)

            if len(smooth_indices) > 0:

                coords_d = smooth_indices // (h_patches * w_patches)
                coords_hw = smooth_indices % (h_patches * w_patches)
                coords_h = coords_hw // w_patches
                coords_w = coords_hw % w_patches
                
                smooth_coords = torch.stack([coords_d, coords_h, coords_w], dim=1)  # [N_smooth, 3]

                used = torch.zeros(len(smooth_coords), dtype=torch.bool, device=x.device)
                
                for i in range(len(smooth_coords)):
                    if used[i]:
                        continue
                    
                    current_coord = smooth_coords[i]

                    neighbors = [i]
                    
                    for j in range(i + 1, len(smooth_coords)):
                        if used[j]:
                            continue
                        neighbor_coord = smooth_coords[j]

                        dist = torch.abs(current_coord - neighbor_coord)
                        if torch.all(dist <= 1):
                            neighbors.append(j)

                    if len(neighbors) >= 4:

                        merge_group = neighbors[:min(8, len(neighbors))]
                        used[merge_group] = True

                        merge_coords = smooth_coords[merge_group]
                        
                        # 计算合并区域的起始和结束位置
                        coord_min = merge_coords.min(dim=0)[0]
                        coord_max = merge_coords.max(dim=0)[0]

                        merge_patch_indices = smooth_indices[merge_group]
                        merged_patches = patches[b, merge_patch_indices]  # [N_merge, C, p, p, p]

                        if len(merge_group) >= 4:

                            avg_patch = merged_patches.mean(dim=0, keepdim=True)  # [1, C, p, p, p]
                            merged_token = self.patch_embed(avg_patch)  # [1, hidden_size, 1, 1, 1]
                            merged_token = merged_token.view(1, self.hidden_size)
                            batch_tokens.append(merged_token)
                        else:

                            for idx in merge_group:
                                single_patch = patches[b:b+1, smooth_indices[idx]:smooth_indices[idx]+1]
                                single_patch = single_patch[0]
                                single_token = self.patch_embed(single_patch)
                                single_token = single_token.view(1, self.hidden_size)
                                batch_tokens.append(single_token)
                    else:

                        single_patch = patches[b:b+1, smooth_indices[i]:smooth_indices[i]+1]
                        single_patch = single_patch[0]
                        single_token = self.patch_embed(single_patch)
                        single_token = single_token.view(1, self.hidden_size)
                        batch_tokens.append(single_token)
            
            if batch_tokens:
                batch_tokens = torch.cat(batch_tokens, dim=0)  # [num_tokens, hidden_size]
                tokens_list.append(batch_tokens)
            else:

                tokens_list.append(torch.zeros(1, self.hidden_size, device=x.device))

        max_tokens = max(t.shape[0] for t in tokens_list)
        tokens_padded = torch.zeros(B, max_tokens, self.hidden_size, device=x.device)
        mask_tokens = torch.zeros(B, max_tokens, device=x.device)
        
        for b, tokens in enumerate(tokens_list):
            num_tokens = tokens.shape[0]
            tokens_padded[b, :num_tokens] = tokens
            mask_tokens[b, :num_tokens] = 1

        pos_embed = self.get_positional_encoding((d_patches, h_patches, w_patches), x.device)
        tokens_padded = tokens_padded + pos_embed[:, :max_tokens]
        
        return tokens_padded, mask_tokens


class Block(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.gate_proj = nn.Linear(hidden_size, num_heads)

    def forward(self, x, h):
        # x: [B, L, C], h: [B, L, C]
        gate_scores = torch.sigmoid(self.gate_proj(h))  # [B, L, num_heads]
        
        attn_output = self.attn(self.norm1(x))  # [B, L, C]
        
        B, L, D = attn_output.shape
        head_dim = D // self.num_heads
        attn_output_heads = attn_output.view(B, L, self.num_heads, head_dim)
        
        # 应用门控机制
        gated_attn = attn_output_heads * gate_scores.unsqueeze(-1).view(B, L, self.num_heads, 1)
        gated_attn = gated_attn.view(B, L, D)
        
        x = x + gated_attn
        x = x + self.mlp(self.norm2(x))
        
        return x


class UNET(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            input_shape=(64, 64, 32),
            hidden_size=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            patch_size=4,
            mask_threshold=0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.mask_threshold = mask_threshold
        
        D, H, W = input_shape
        self.d_patches = D // patch_size
        self.h_patches = H // patch_size
        self.w_patches = W // patch_size
        self.num_patches_total = self.d_patches * self.h_patches * self.w_patches

        self.adaptive_embed = AdaptiveBlockEmbedding(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size
        )

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, patch_size ** 3 * out_channels)
        )

        self.reconstruct_conv = nn.ConvTranspose3d(
            out_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.xavier_uniform_(self.output_proj[1].weight)
        nn.init.xavier_uniform_(self.output_proj[3].weight)

    def forward(self, x):

        B, C, D, H, W = x.shape
        
        # Step 1: 生成八叉树掩码
        with torch.no_grad():
            mask = generate_octree_masks(
                x, 
                s=self.mask_threshold,
                D_m=self.d_patches,
                H_m=self.h_patches,
                W_m=self.w_patches
            )

        tokens, token_mask = self.adaptive_embed(x, mask)  # [B, num_tokens, hidden_size]
        
        # Step 3: Transformer处理
        h = tokens
        for block in self.blocks:
            tokens = block(tokens, h)

        patches_flat = self.output_proj(tokens)  # [B, num_tokens, patch_size^3 * out_channels]

        p = self.patch_size
        patches_reshaped = patches_flat.view(B, -1, self.out_channels, p, p, p)

        if patches_reshaped.shape[1] == self.num_patches_total:
            # [B, d_patches*h_patches*w_patches, C, p, p, p]
            # -> [B, C, d_patches, h_patches, w_patches, p, p, p]
            patches_grid = patches_reshaped.view(
                B, self.d_patches, self.h_patches, self.w_patches, 
                self.out_channels, p, p, p
            )
            # -> [B, C, d_patches*p, h_patches*p, w_patches*p]
            # -> [B, C, D, H, W]
            patches_grid = patches_grid.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            output = patches_grid.view(B, self.out_channels, D, H, W)
        else:
            output = torch.zeros(B, self.out_channels, D, H, W, device=x.device)

        return output


if __name__ == "__main__":
    print("=" * 70)
    print("测试自适应Mask UNET模型")
    print("=" * 70)
    
    # 创建模型
    model = UNET(
        in_channels=1,
        out_channels=1,
        input_shape=(64, 64, 32),
        hidden_size=384,
        depth=12,
        num_heads=6,
        patch_size=2,
        mask_threshold=0
    )
    
    print(f"\n模型结构:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建测试数据
    batch_size = 1
    D, H, W = 64, 64, 32
    x = torch.randn(batch_size, 1, D, H, W)
    
    print(f"\n输入形状: {x.shape}")
    print(f"期望输出形状: ({batch_size}, 1, {D}, {H}, {W})")
    
    try:
        with torch.no_grad():
            output = model(x)
        print(f"\n实际输出形状: {output.shape}")
        
        if output.shape == (batch_size, 1, D, H, W):
            print("\n✓ 模型前向传播成功！输出形状符合预期。")
        else:
            print(f"\n⚠ 输出形状不符合预期！")
        
        print(f"\n输出统计:")
        print(f"  均值: {output.mean().item():.6f}")
        print(f"  标准差: {output.std().item():.6f}")
        print(f"  最小值: {output.min().item():.6f}")
        print(f"  最大值: {output.max().item():.6f}")
        
    except Exception as e:
        print(f"\n✗ 模型前向传播失败！")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
