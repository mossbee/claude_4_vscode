"""
Enhanced ResNet backbone with attention mechanisms for twin face verification.
"""

import torch
import torch.nn as nn
import timm
from .attention import CBAM, NonLocalBlock


class AttentionResNet(nn.Module):
    """ResNet backbone enhanced with attention mechanisms."""
    
    def __init__(self, 
                 model_name='resnet50',
                 pretrained=True,
                 num_classes=0,
                 add_cbam=True,
                 add_nonlocal=True,
                 nonlocal_stages=[2, 3]):
        super(AttentionResNet, self).__init__()
        
        # Load base ResNet model
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
        # Get the feature dimension
        if hasattr(self.backbone, 'num_features'):
            self.feat_dim = self.backbone.num_features
        else:
            # For ResNet50, this is typically 2048
            self.feat_dim = 2048
        
        self.add_cbam = add_cbam
        self.add_nonlocal = add_nonlocal
        self.nonlocal_stages = nonlocal_stages
        
        # Add attention modules to different stages
        if add_cbam:
            self._add_cbam_modules()
        
        if add_nonlocal:
            self._add_nonlocal_modules()
    
    def _add_cbam_modules(self):
        """Add CBAM modules to ResNet layers."""
        # Add CBAM to each residual stage
        self.cbam_modules = nn.ModuleDict()
        
        # Get channel dimensions for each stage
        stage_channels = {
            'layer1': 256,   # ResNet50 layer1 output channels
            'layer2': 512,   # ResNet50 layer2 output channels  
            'layer3': 1024,  # ResNet50 layer3 output channels
            'layer4': 2048   # ResNet50 layer4 output channels
        }
        
        for stage_name, channels in stage_channels.items():
            if hasattr(self.backbone, stage_name):
                self.cbam_modules[stage_name] = CBAM(channels)
    
    def _add_nonlocal_modules(self):
        """Add Non-local blocks to specified stages."""
        self.nonlocal_modules = nn.ModuleDict()
        
        stage_channels = {
            'layer1': 256,
            'layer2': 512, 
            'layer3': 1024,
            'layer4': 2048
        }
        
        for stage_idx in self.nonlocal_stages:
            stage_name = f'layer{stage_idx}'
            if stage_name in stage_channels:
                channels = stage_channels[stage_name]
                self.nonlocal_modules[stage_name] = NonLocalBlock(channels)
    
    def forward(self, x):
        """
        Forward pass with attention enhancement.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            features: Output features [B, feat_dim, H', W']
        """
        # Initial convolution and pooling
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Process through ResNet layers with attention
        stage_outputs = {}
        
        # Layer 1
        x = self.backbone.layer1(x)
        if self.add_cbam and 'layer1' in self.cbam_modules:
            x = self.cbam_modules['layer1'](x)
        if self.add_nonlocal and 'layer1' in self.nonlocal_modules:
            x = self.nonlocal_modules['layer1'](x)
        stage_outputs['layer1'] = x
        
        # Layer 2
        x = self.backbone.layer2(x)
        if self.add_cbam and 'layer2' in self.cbam_modules:
            x = self.cbam_modules['layer2'](x)
        if self.add_nonlocal and 'layer2' in self.nonlocal_modules:
            x = self.nonlocal_modules['layer2'](x)
        stage_outputs['layer2'] = x
        
        # Layer 3
        x = self.backbone.layer3(x)
        if self.add_cbam and 'layer3' in self.cbam_modules:
            x = self.cbam_modules['layer3'](x)
        if self.add_nonlocal and 'layer3' in self.nonlocal_modules:
            x = self.nonlocal_modules['layer3'](x)
        stage_outputs['layer3'] = x
        
        # Layer 4
        x = self.backbone.layer4(x)
        if self.add_cbam and 'layer4' in self.cbam_modules:
            x = self.cbam_modules['layer4'](x)
        if self.add_nonlocal and 'layer4' in self.nonlocal_modules:
            x = self.nonlocal_modules['layer4'](x)
        stage_outputs['layer4'] = x
        
        return x, stage_outputs
    
    def get_feature_dim(self):
        """Return the feature dimension of the backbone."""
        return self.feat_dim


class MultiScaleBackbone(nn.Module):
    """Multi-scale feature extraction backbone."""
    
    def __init__(self, 
                 model_name='resnet50',
                 pretrained=True,
                 scales=[1, 0.75, 0.5]):
        super(MultiScaleBackbone, self).__init__()
        
        self.scales = scales
        self.backbones = nn.ModuleList([
            AttentionResNet(model_name, pretrained, num_classes=0)
            for _ in scales
        ])
        
        # Feature fusion
        feat_dim = self.backbones[0].get_feature_dim()
        self.fusion = nn.Conv2d(
            feat_dim * len(scales), 
            feat_dim, 
            kernel_size=1, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(feat_dim)
        
    def forward(self, x):
        """
        Multi-scale forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Fused multi-scale features
        """
        B, C, H, W = x.shape
        scale_features = []
        
        for i, scale in enumerate(self.scales):
            if scale != 1.0:
                # Resize input
                scale_h, scale_w = int(H * scale), int(W * scale)
                x_scaled = torch.nn.functional.interpolate(
                    x, size=(scale_h, scale_w), 
                    mode='bilinear', align_corners=False
                )
            else:
                x_scaled = x
            
            # Extract features
            features, _ = self.backbones[i](x_scaled)
            
            # Resize features back to original spatial size
            if scale != 1.0:
                features = torch.nn.functional.interpolate(
                    features, size=features.shape[-2:], 
                    mode='bilinear', align_corners=False
                )
            
            scale_features.append(features)
        
        # Concatenate and fuse multi-scale features
        fused = torch.cat(scale_features, dim=1)
        fused = self.bn(self.fusion(fused))
        
        return fused


class GradientCheckpointBackbone(AttentionResNet):
    """ResNet backbone with gradient checkpointing for memory efficiency."""
    
    def __init__(self, *args, **kwargs):
        super(GradientCheckpointBackbone, self).__init__(*args, **kwargs)
        self.use_checkpointing = True
    
    def forward(self, x):
        """Forward pass with gradient checkpointing."""
        if self.training and self.use_checkpointing:
            return self._forward_with_checkpointing(x)
        else:
            return super().forward(x)
    
    def _forward_with_checkpointing(self, x):
        """Forward pass using gradient checkpointing."""
        from torch.utils.checkpoint import checkpoint
        
        # Initial layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Checkpoint each layer
        x = checkpoint(self._forward_layer1, x)
        x = checkpoint(self._forward_layer2, x)
        x = checkpoint(self._forward_layer3, x)
        x = checkpoint(self._forward_layer4, x)
        
        return x, {}
    
    def _forward_layer1(self, x):
        x = self.backbone.layer1(x)
        if self.add_cbam and 'layer1' in self.cbam_modules:
            x = self.cbam_modules['layer1'](x)
        if self.add_nonlocal and 'layer1' in self.nonlocal_modules:
            x = self.nonlocal_modules['layer1'](x)
        return x
    
    def _forward_layer2(self, x):
        x = self.backbone.layer2(x)
        if self.add_cbam and 'layer2' in self.cbam_modules:
            x = self.cbam_modules['layer2'](x)
        if self.add_nonlocal and 'layer2' in self.nonlocal_modules:
            x = self.nonlocal_modules['layer2'](x)
        return x
    
    def _forward_layer3(self, x):
        x = self.backbone.layer3(x)
        if self.add_cbam and 'layer3' in self.cbam_modules:
            x = self.cbam_modules['layer3'](x)
        if self.add_nonlocal and 'layer3' in self.nonlocal_modules:
            x = self.nonlocal_modules['layer3'](x)
        return x
    
    def _forward_layer4(self, x):
        x = self.backbone.layer4(x)
        if self.add_cbam and 'layer4' in self.cbam_modules:
            x = self.cbam_modules['layer4'](x)
        if self.add_nonlocal and 'layer4' in self.nonlocal_modules:
            x = self.nonlocal_modules['layer4'](x)
        return x
