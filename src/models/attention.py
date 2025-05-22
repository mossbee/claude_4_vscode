"""
Attention mechanisms for twin face verification.
Implements SE, CBAM, Non-local, and Cross-attention modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module from CBAM."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class NonLocalBlock(nn.Module):
    """Non-local attention block for capturing long-range dependencies."""
    
    def __init__(self, in_channels, reduction=2):
        super(NonLocalBlock, self).__init__()
        self.inter_channels = in_channels // reduction
        
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
        # Initialize W to zero for stable training
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Compute query, key, value
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        
        # Compute attention
        attention = torch.matmul(theta_x, phi_x)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        y = torch.matmul(attention, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, H, W)
        
        # Final projection and residual connection
        W_y = self.W(y)
        return x + W_y


class CrossAttention(nn.Module):
    """Cross-attention module for comparing features between two images."""
    
    def __init__(self, dim, heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        """
        Args:
            x: Query features [B, N, C]
            y: Key-Value features [B, M, C]
        Returns:
            Cross-attended features [B, N, C]
        """
        B, N, C = x.shape
        _, M, _ = y.shape
        
        # Project to query, key, value
        q = self.q_proj(x).reshape(B, N, self.heads, self.dim_head).transpose(1, 2)  # [B, H, N, D]
        k = self.k_proj(y).reshape(B, M, self.heads, self.dim_head).transpose(1, 2)  # [B, H, M, D]
        v = self.v_proj(y).reshape(B, M, self.heads, self.dim_head).transpose(1, 2)  # [B, H, M, D]
        
        # Scaled dot-product attention
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, v)  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        return self.out_proj(out)


class MultiScaleCrossAttention(nn.Module):
    """Multi-scale cross-attention for capturing different levels of detail."""
    
    def __init__(self, dim, heads=8, scales=[1, 2, 4], dropout=0.1):
        super(MultiScaleCrossAttention, self).__init__()
        self.scales = scales
        self.cross_attentions = nn.ModuleList([
            CrossAttention(dim, heads, dropout) for _ in scales
        ])
        self.fusion = nn.Linear(dim * len(scales), dim)
        
    def forward(self, x, y):
        """
        Args:
            x: Query features [B, N, C]
            y: Key-Value features [B, M, C]
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))  # Assume square feature maps
        
        outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                # Original resolution
                att_out = self.cross_attentions[i](x, y)
            else:
                # Downsample for multi-scale processing
                x_scaled = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
                y_scaled = y.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
                
                x_scaled = F.adaptive_avg_pool2d(x_scaled, (H//scale, W//scale))
                y_scaled = F.adaptive_avg_pool2d(y_scaled, (H//scale, W//scale))
                
                x_scaled = x_scaled.permute(0, 2, 3, 1).view(B, -1, C)
                y_scaled = y_scaled.permute(0, 2, 3, 1).view(B, -1, C)
                
                att_out = self.cross_attentions[i](x_scaled, y_scaled)
                
                # Upsample back to original resolution
                att_out = att_out.view(B, H//scale, W//scale, C).permute(0, 3, 1, 2)
                att_out = F.interpolate(att_out, size=(H, W), mode='bilinear', align_corners=False)
                att_out = att_out.permute(0, 2, 3, 1).view(B, N, C)
            
            outputs.append(att_out)
        
        # Fuse multi-scale features
        fused = torch.cat(outputs, dim=-1)
        return self.fusion(fused)


class AttentionFusion(nn.Module):
    """Fusion module for combining original and cross-attended features."""
    
    def __init__(self, dim):
        super(AttentionFusion, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, original, cross_attended):
        """
        Args:
            original: Original features [B, N, C]
            cross_attended: Cross-attended features [B, N, C]
        """
        original = self.norm1(original)
        cross_attended = self.norm2(cross_attended)
        
        # Compute gating weights
        combined = torch.cat([original, cross_attended], dim=-1)
        gate_weights = self.gate(combined)
        
        # Weighted combination
        fused = gate_weights * original + (1 - gate_weights) * cross_attended
        return fused
