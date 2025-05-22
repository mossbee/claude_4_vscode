"""
Twin Verifier: Main model for twin face verification.
Implements Siamese network with attention mechanisms and twin-aware loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import AttentionResNet, GradientCheckpointBackbone
from .attention import CrossAttention, MultiScaleCrossAttention, AttentionFusion


class TwinVerifier(nn.Module):
    """
    Twin face verification model with Siamese architecture and attention mechanisms.
    """
    
    def __init__(self, 
                 backbone='resnet50',
                 feat_dim=512,
                 num_heads=8,
                 dropout=0.1,
                 use_gradient_checkpointing=True,
                 use_multi_scale_attention=False):
        super(TwinVerifier, self).__init__()
        
        # Feature extraction backbone
        if use_gradient_checkpointing:
            self.backbone = GradientCheckpointBackbone(
                model_name=backbone,
                pretrained=True,
                num_classes=0,
                add_cbam=True,
                add_nonlocal=True
            )
        else:
            self.backbone = AttentionResNet(
                model_name=backbone,
                pretrained=True,
                num_classes=0,
                add_cbam=True,
                add_nonlocal=True
            )
        
        self.backbone_dim = self.backbone.get_feature_dim()
        self.feat_dim = feat_dim
        
        # Cross-attention module
        if use_multi_scale_attention:
            self.cross_attention = MultiScaleCrossAttention(
                self.backbone_dim, num_heads, dropout=dropout
            )
        else:
            self.cross_attention = CrossAttention(
                self.backbone_dim, num_heads, dropout=dropout
            )
        
        # Attention fusion
        self.attention_fusion = AttentionFusion(self.backbone_dim)
        
        # Feature projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim * 3, self.backbone_dim),
            nn.BatchNorm1d(self.backbone_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.backbone_dim, feat_dim)
        )
        
        # Difference head for auxiliary classification
        self.difference_head = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 3)  # Same/Twin/Other
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward_once(self, x):
        """
        Forward pass for a single image.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            features: Feature tokens [B, N, C] where N = H*W
        """
        # Extract features using backbone
        features, _ = self.backbone(x)  # [B, C, H, W]
        
        # Convert to tokens for attention
        B, C, H, W = features.shape
        tokens = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        return tokens
    
    def forward(self, x1, x2=None, return_attention=False):
        """
        Forward pass for twin verification.
        
        Args:
            x1: First image [B, C, H, W]
            x2: Second image [B, C, H, W] (optional for single image encoding)
            return_attention: Whether to return attention maps
            
        Returns:
            If x2 is None: Single image embedding
            If x2 is provided: (embedding1, embedding2, difference_logits, attention_maps)
        """
        # Extract features for first image
        tokens1 = self.forward_once(x1)  # [B, N, C]
        
        if x2 is None:
            # Single image encoding
            pooled = tokens1.mean(dim=1)  # Global average pooling
            embedding = self.projection(
                torch.cat([pooled, pooled, torch.zeros_like(pooled)], dim=1)
            )
            return F.normalize(embedding, p=2, dim=1)
        
        # Extract features for second image
        tokens2 = self.forward_once(x2)  # [B, N, C]
        
        # Cross-attention between the two images
        cross_attended1 = self.cross_attention(tokens1, tokens2)  # What parts of x2 explain x1
        cross_attended2 = self.cross_attention(tokens2, tokens1)  # What parts of x1 explain x2
        
        # Fuse original and cross-attended features
        fused1 = self.attention_fusion(tokens1, cross_attended1)
        fused2 = self.attention_fusion(tokens2, cross_attended2)
        
        # Global pooling
        pooled1 = fused1.mean(dim=1)  # [B, C]
        pooled2 = fused2.mean(dim=1)  # [B, C]
        pooled_cross = (cross_attended1.mean(dim=1) + cross_attended2.mean(dim=1)) / 2
        
        # Compute difference features
        diff_features = torch.abs(pooled1 - pooled2)
        
        # Project to final embeddings
        combined1 = torch.cat([pooled1, pooled_cross, diff_features], dim=1)
        combined2 = torch.cat([pooled2, pooled_cross, diff_features], dim=1)
        
        embedding1 = self.projection(combined1)
        embedding2 = self.projection(combined2)
        
        # Normalize embeddings
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        embedding2 = F.normalize(embedding2, p=2, dim=1)
        
        # Difference head for auxiliary classification
        diff_input = torch.cat([embedding1, embedding2, torch.abs(embedding1 - embedding2)], dim=1)
        difference_logits = self.difference_head(diff_input)
        
        outputs = {
            'embedding1': embedding1,
            'embedding2': embedding2,
            'difference_logits': difference_logits
        }
        
        if return_attention:
            # Compute attention maps for visualization
            attention_maps = self._compute_attention_maps(tokens1, tokens2, cross_attended1, cross_attended2)
            outputs['attention_maps'] = attention_maps
        
        return outputs
    
    def _compute_attention_maps(self, tokens1, tokens2, cross_attended1, cross_attended2):
        """
        Compute attention maps for visualization.
        
        Args:
            tokens1, tokens2: Original feature tokens
            cross_attended1, cross_attended2: Cross-attended features
            
        Returns:
            Dictionary of attention maps
        """
        B, N, C = tokens1.shape
        H = W = int(N ** 0.5)  # Assume square feature maps
        
        # Compute attention weights (simplified visualization)
        attention1 = torch.norm(cross_attended1 - tokens1, dim=2)  # [B, N]
        attention2 = torch.norm(cross_attended2 - tokens2, dim=2)  # [B, N]
        
        # Reshape to spatial dimensions
        attention1 = attention1.view(B, H, W)
        attention2 = attention2.view(B, H, W)
        
        return {
            'attention1': attention1,
            'attention2': attention2
        }
    
    def compute_similarity(self, embedding1, embedding2, metric='cosine'):
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1, embedding2: Feature embeddings
            metric: 'cosine' or 'l2'
            
        Returns:
            Similarity scores
        """
        if metric == 'cosine':
            return F.cosine_similarity(embedding1, embedding2, dim=1)
        elif metric == 'l2':
            return -torch.norm(embedding1 - embedding2, p=2, dim=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")


class LightweightTwinVerifier(TwinVerifier):
    """Lightweight version of TwinVerifier for faster inference."""
    
    def __init__(self, *args, **kwargs):
        kwargs['backbone'] = kwargs.get('backbone', 'resnet18')
        kwargs['feat_dim'] = kwargs.get('feat_dim', 256)
        kwargs['use_multi_scale_attention'] = False
        super(LightweightTwinVerifier, self).__init__(*args, **kwargs)
        
        # Simpler projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim * 3, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        
        # Simpler difference head
        self.difference_head = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 3)
        )


class EnsembleTwinVerifier(nn.Module):
    """Ensemble of multiple TwinVerifier models for improved performance."""
    
    def __init__(self, model_configs, ensemble_method='average'):
        super(EnsembleTwinVerifier, self).__init__()
        
        self.models = nn.ModuleList([
            TwinVerifier(**config) for config in model_configs
        ])
        self.ensemble_method = ensemble_method
        self.num_models = len(self.models)
        
        if ensemble_method == 'learned':
            # Learnable ensemble weights
            feat_dim = model_configs[0].get('feat_dim', 512)
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
            self.fusion_layer = nn.Linear(feat_dim * self.num_models, feat_dim)
    
    def forward(self, x1, x2=None):
        """Ensemble forward pass."""
        if x2 is None:
            # Single image encoding
            embeddings = []
            for model in self.models:
                emb = model(x1)
                embeddings.append(emb)
            return self._ensemble_embeddings(embeddings)
        
        # Paired image processing
        outputs = []
        for model in self.models:
            out = model(x1, x2)
            outputs.append(out)
        
        return self._ensemble_outputs(outputs)
    
    def _ensemble_embeddings(self, embeddings):
        """Ensemble single embeddings."""
        if self.ensemble_method == 'average':
            return torch.stack(embeddings).mean(dim=0)
        elif self.ensemble_method == 'learned':
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted = sum(w * emb for w, emb in zip(weights, embeddings))
            return weighted
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _ensemble_outputs(self, outputs):
        """Ensemble paired outputs."""
        # Average embeddings and difference logits
        embedding1 = torch.stack([out['embedding1'] for out in outputs]).mean(dim=0)
        embedding2 = torch.stack([out['embedding2'] for out in outputs]).mean(dim=0)
        difference_logits = torch.stack([out['difference_logits'] for out in outputs]).mean(dim=0)
        
        return {
            'embedding1': embedding1,
            'embedding2': embedding2,
            'difference_logits': difference_logits
        }
