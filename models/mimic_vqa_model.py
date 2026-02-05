"""
MIMIC-CXR VQA Model

Complete SSG-VQA-Net adapted for chest X-ray VQA with:
- ConvNeXt-Base visual backbone
- Bio+ClinicalBERT text encoder
- Expanded scene graph encoding (134 dims)
- Multi-head answer module
- CheXpert auxiliary classification

Based on MIMIC_CXR_VQA_ANALYSIS.md specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class MIMICVQAOutput(dict):
    """Output container for MIMIC VQA model that behaves like a dict and supports attribute access."""

    def __init__(self, vqa_logits, chexpert_logits, pooled_output, hidden_states=None):
        super().__init__(
            vqa_logits=vqa_logits,
            chexpert_logits=chexpert_logits,
            pooled_output=pooled_output,
            hidden_states=hidden_states,
        )
        # Mirror dict keys as attributes for compatibility
        for k, v in self.items():
            object.__setattr__(self, k, v)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Keep dict and attribute state in sync
        self[name] = value
        object.__setattr__(self, name, value)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return super().keys()

    def values(self):
        return super().values()

    def items(self):
        return super().items()

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        object.__setattr__(self, key, value)

    def __delitem__(self, key):
        super().__delitem__(key)
        object.__delattr__(self, key)



# Single dict-like MIMICVQAOutput class is defined above; removed duplicate dataclass.

class ConvNeXtFeatureExtractor(nn.Module):
    """
    ConvNeXt-Base visual backbone for chest X-ray images.
    Outputs 512-dim features to match SSG-VQA format.
    """
    
    def __init__(
        self,
        model_name: str = 'convnext_base',
        pretrained: bool = True,
        output_dim: int = 512
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        if TIMM_AVAILABLE:
            # Load ConvNeXt from timm
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=[-1]  # Last feature map
            )
            
            # Get output channels from backbone
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feats = self.backbone(dummy)
                backbone_dim = feats[-1].shape[1]
        else:
            # Fallback to simple conv backbone
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            backbone_dim = 512
        
        # Projection to output_dim
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # ROI pooling for region features
        self.roi_pool_size = 7
    
    def forward(
        self,
        images: torch.Tensor,
        bboxes: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Extract visual features.
        
        Args:
            images: (B, 3, H, W) images
            bboxes: Optional list of (N_i, 4) bbox tensors per image
            
        Returns:
            features: (B, output_dim) global features, or
                     (B, max_N, output_dim) if bboxes provided
        """
        # Extract feature maps
        if TIMM_AVAILABLE:
            feature_maps = self.backbone(images)[-1]  # (B, C, H', W')
        else:
            feature_maps = self.backbone(images)
        
        if bboxes is None:
            # Global features only
            return self.projection(feature_maps)
        
        # ROI features
        batch_size = images.shape[0]
        device = images.device
        
        # Find max objects
        max_objects = max(len(b) for b in bboxes) if bboxes else 1
        
        # Extract ROI features for each image
        all_features = []
        
        for b in range(batch_size):
            if b < len(bboxes) and len(bboxes[b]) > 0:
                img_bboxes = bboxes[b]  # (N, 4)
                num_objects = len(img_bboxes)
                
                # Extract ROI features using simple crop and resize
                roi_features = []
                for box in img_bboxes:
                    # Convert normalized coords to feature map coords
                    h, w = feature_maps.shape[2:]
                    x1 = int(box[0] * w)
                    y1 = int(box[1] * h)
                    x2 = int(box[2] * w)
                    y2 = int(box[3] * h)
                    
                    # Ensure valid bbox
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    if x2 <= x1:
                        x2 = x1 + 1
                    if y2 <= y1:
                        y2 = y1 + 1
                    
                    # Crop and pool
                    roi = feature_maps[b:b+1, :, y1:y2, x1:x2]
                    pooled = F.adaptive_avg_pool2d(roi, (1, 1))
                    roi_features.append(pooled)
                
                roi_features = torch.cat(roi_features, dim=0)  # (N, C, 1, 1)
                roi_features = roi_features.flatten(1)  # (N, C)
                
                # Project
                roi_features = self.projection[2:](roi_features)  # Skip pooling
                
                # Pad to max_objects
                if num_objects < max_objects:
                    padding = torch.zeros(
                        max_objects - num_objects,
                        self.output_dim,
                        device=device
                    )
                    roi_features = torch.cat([roi_features, padding], dim=0)
                
                all_features.append(roi_features)
            else:
                # No bboxes, use global feature
                global_feat = self.projection(feature_maps[b:b+1])  # (1, D)
                padding = torch.zeros(max_objects - 1, self.output_dim, device=device)
                all_features.append(torch.cat([global_feat, padding], dim=0))
        
        return torch.stack(all_features)  # (B, max_N, D)


class SceneGraphEncoder(nn.Module):
    """
    Encodes scene graph information into 134-dim features.
    
    Feature breakdown:
    - bbox_coords: 4 dims (normalized x1, y1, x2, y2)
    - bbox_area: 1 dim
    - aspect_ratio: 1 dim
    - region_embedding: 64 dims
    - entity_embedding: 64 dims
    Total: 134 dims
    """
    
    def __init__(
        self,
        num_regions: int = 310,
        num_entities: int = 237,
        embedding_dim: int = 64
    ):
        super().__init__()
        
        self.num_regions = num_regions
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.output_dim = 6 + embedding_dim * 2  # 134
        
        # Learnable embeddings
        self.region_embedding = nn.Embedding(num_regions + 1, embedding_dim)  # +1 for unknown
        self.entity_embedding = nn.Embedding(num_entities + 1, embedding_dim)
        
        # Aggregation layers
        self.region_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        self.entity_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
    
    def forward(
        self,
        scene_graphs: List[Dict[str, Any]],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode scene graphs.
        
        Args:
            scene_graphs: List of scene graph dicts with:
                - bboxes: (N, 4) normalized coordinates
                - region_ids: (N,) region indices
                - entity_ids: (N,) entity indices
            device: Target device
            
        Returns:
            features: (B, max_N, 134) scene graph features
            mask: (B, max_N) attention mask
        """
        batch_size = len(scene_graphs)
        max_objects = max(sg['num_objects'] for sg in scene_graphs) if scene_graphs else 1
        
        all_features = []
        all_masks = []
        
        for sg in scene_graphs:
            num_objects = sg['num_objects']
            bboxes = torch.tensor(sg['bboxes'], dtype=torch.float, device=device)
            region_ids = torch.tensor(sg['region_ids'], dtype=torch.long, device=device)
            entity_ids = torch.tensor(sg['entity_ids'], dtype=torch.long, device=device)
            
            # Bbox features (6 dims)
            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect = w / (h + 1e-6)
            bbox_features = torch.stack([x1, y1, x2, y2, area, aspect], dim=1)  # (N, 6)
            
            # Region embeddings (64 dims)
            region_emb = self.region_embedding(region_ids)  # (N, 64)
            region_emb = self.region_proj(region_emb)
            
            # Entity embeddings (64 dims)
            entity_emb = self.entity_embedding(entity_ids)  # (N, 64)
            entity_emb = self.entity_proj(entity_emb)
            
            # Combine (134 dims)
            features = torch.cat([bbox_features, region_emb, entity_emb], dim=1)  # (N, 134)
            
            # Create mask
            mask = torch.ones(num_objects, device=device)
            
            # Pad to max_objects
            if num_objects < max_objects:
                feat_padding = torch.zeros(max_objects - num_objects, self.output_dim, device=device)
                features = torch.cat([features, feat_padding], dim=0)
                mask_padding = torch.zeros(max_objects - num_objects, device=device)
                mask = torch.cat([mask, mask_padding], dim=0)
            
            all_features.append(features)
            all_masks.append(mask)
        
        return torch.stack(all_features), torch.stack(all_masks)


class TextEncoder(nn.Module):
    """
    Bio+ClinicalBERT text encoder for medical questions.
    """
    
    def __init__(
        self,
        model_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
        output_dim: int = 768,
        freeze_layers: int = 0
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        if TRANSFORMERS_AVAILABLE:
            self.encoder = AutoModel.from_pretrained(model_name)
            
            # Freeze bottom layers if specified
            if freeze_layers > 0:
                for param in self.encoder.embeddings.parameters():
                    param.requires_grad = False
                for layer in self.encoder.encoder.layer[:freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            # Fallback to simple embedding
            self.encoder = None
            self.embedding = nn.Embedding(30522, output_dim)
            self.lstm = nn.LSTM(output_dim, output_dim // 2, bidirectional=True, batch_first=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text.
        
        Returns:
            sequence_output: (B, L, D) token embeddings
            pooled_output: (B, D) [CLS] representation
        """
        if self.encoder is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )
            return outputs.last_hidden_state, outputs.pooler_output
        else:
            # Fallback
            embeds = self.embedding(input_ids)
            output, (h, c) = self.lstm(embeds)
            pooled = torch.cat([h[-2], h[-1]], dim=1)
            return output, pooled


class SceneEmbeddedInteraction(nn.Module):
    """
    Scene-Embedded Interaction Module (SIM).
    
    Fuses visual, scene graph, and text features through
    cross-attention layers.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Cross-attention layers
        self.visual_to_text = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.text_to_visual = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.scene_to_text = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.visual_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.text_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.scene_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        scene_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse multimodal features.
        
        Args:
            visual_features: (B, N_v, D) visual tokens
            text_features: (B, N_t, D) text tokens
            scene_features: (B, N_s, D) scene graph features
            *_mask: Attention masks (1=attend, 0=ignore)
            
        Returns:
            fused: (B, D) fused representation
        """
        # Convert masks to attention format if provided
        visual_key_padding_mask = ~visual_mask.bool() if visual_mask is not None else None
        text_key_padding_mask = ~text_mask.bool() if text_mask is not None else None
        scene_key_padding_mask = ~scene_mask.bool() if scene_mask is not None else None
        
        for i in range(self.num_layers):
            # Visual attends to text
            visual_attended, _ = self.visual_to_text[i](
                visual_features, text_features, text_features,
                key_padding_mask=text_key_padding_mask
            )
            visual_features = self.visual_norms[i](visual_features + visual_attended)
            
            # Text attends to visual
            text_attended, _ = self.text_to_visual[i](
                text_features, visual_features, visual_features,
                key_padding_mask=visual_key_padding_mask
            )
            text_features = self.text_norms[i](text_features + text_attended)
            
            # Scene attends to text
            scene_attended, _ = self.scene_to_text[i](
                scene_features, text_features, text_features,
                key_padding_mask=text_key_padding_mask
            )
            scene_features = self.scene_norms[i](scene_features + scene_attended)
        
        # Combine all modalities
        # Pool each modality
        if visual_mask is not None:
            visual_pooled = (visual_features * visual_mask.unsqueeze(-1)).sum(1) / visual_mask.sum(1, keepdim=True).clamp(min=1)
        else:
            visual_pooled = visual_features.mean(1)
        
        if text_mask is not None:
            text_pooled = (text_features * text_mask.unsqueeze(-1)).sum(1) / text_mask.sum(1, keepdim=True).clamp(min=1)
        else:
            text_pooled = text_features.mean(1)
        
        if scene_mask is not None:
            scene_pooled = (scene_features * scene_mask.unsqueeze(-1)).sum(1) / scene_mask.sum(1, keepdim=True).clamp(min=1)
        else:
            scene_pooled = scene_features.mean(1)
        
        # Combine and project
        combined = visual_pooled + text_pooled + scene_pooled
        output = self.ffn_norm(combined + self.ffn(combined))
        
        return output


class MultiHeadAnswerModule(nn.Module):
    """
    Multi-head answer module for different question types.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_binary_classes: int = 2,
        num_category_classes: int = 14,
        num_region_classes: int = 26,
        num_severity_classes: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Binary head (Yes/No)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_binary_classes)
        )
        
        # Category head (Finding types)
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_category_classes)
        )
        
        # Region head (Anatomical regions)
        self.region_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_region_classes)
        )
        
        # Severity head
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_severity_classes)
        )
    
    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute logits for all heads.
        
        Args:
            pooled_output: (B, D) fused representation
            
        Returns:
            Dict mapping head names to logits tensors
        """
        return {
            'binary': self.binary_head(pooled_output),
            'category': self.category_head(pooled_output),
            'region': self.region_head(pooled_output),
            'severity': self.severity_head(pooled_output),
        }


class CheXpertHead(nn.Module):
    """
    Auxiliary classification head for CheXpert labels.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 14,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Predict CheXpert labels.
        
        Args:
            pooled_output: (B, D) visual representation
            
        Returns:
            logits: (B, 14) classification logits
        """
        return self.classifier(pooled_output)


class MIMICCXRVQAModel(nn.Module):
    """
    Complete SSG-VQA-Net adapted for MIMIC-CXR VQA.
    
    Architecture:
    1. ConvNeXt-Base visual backbone → 512-dim features
    2. Bio+ClinicalBERT text encoder → 768-dim features
    3. Scene graph encoder → 134-dim features
    4. Feature projection to common dimension
    5. Scene-Embedded Interaction Module
    6. Multi-head answer module
    7. Optional CheXpert auxiliary head
    """
    
    def __init__(
        self,
        visual_backbone: str = 'convnext_base',
        text_encoder: str = 'emilyalsentzer/Bio_ClinicalBERT',
        visual_feature_dim: int = 512,
        scene_graph_dim: int = 134,
        num_regions: int = 310,
        num_entities: int = 237,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        sim_layers: int = 2,
        num_binary_classes: int = 2,
        num_category_classes: int = 14,
        num_region_classes: int = 26,
        num_severity_classes: int = 4,
        dropout: float = 0.1,
        use_chexpert_head: bool = True,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        
        # Visual backbone
        self.visual_encoder = ConvNeXtFeatureExtractor(
            model_name=visual_backbone,
            pretrained=True,
            output_dim=visual_feature_dim
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_encoder,
            output_dim=hidden_size
        )
        
        # Scene graph encoder
        self.scene_encoder = SceneGraphEncoder(
            num_regions=num_regions,
            num_entities=num_entities,
            embedding_dim=64
        )
        
        # Feature projections to hidden_size
        self.visual_proj = nn.Linear(visual_feature_dim, hidden_size)
        self.scene_proj = nn.Linear(scene_graph_dim, hidden_size)
        
        # Scene-Embedded Interaction Module
        self.sim = SceneEmbeddedInteraction(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_layers=sim_layers,
            dropout=dropout
        )
        
        # Multi-head answer module
        self.answer_module = MultiHeadAnswerModule(
            hidden_size=hidden_size,
            num_binary_classes=num_binary_classes,
            num_category_classes=num_category_classes,
            num_region_classes=num_region_classes,
            num_severity_classes=num_severity_classes,
            dropout=dropout
        )
        
        # CheXpert auxiliary head
        self.use_chexpert_head = use_chexpert_head
        if use_chexpert_head:
            self.chexpert_head = CheXpertHead(
                hidden_size=hidden_size,
                num_classes=14,
                dropout=dropout
            )
        else:
            self.chexpert_head = None
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.gradient_checkpointing = True
        if hasattr(self.text_encoder.encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.encoder.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing (fixes DataParallel deadlock)."""
        self.gradient_checkpointing = False
        # Disable in text encoder (BERT)
        if hasattr(self.text_encoder, 'encoder'):
            if hasattr(self.text_encoder.encoder, 'gradient_checkpointing_disable'):
                self.text_encoder.encoder.gradient_checkpointing_disable()
            # Also set the flag directly if it exists
            if hasattr(self.text_encoder.encoder, 'gradient_checkpointing'):
                self.text_encoder.encoder.gradient_checkpointing = False
        # Disable in text encoder config
        if hasattr(self.text_encoder, 'config'):
            self.text_encoder.config.gradient_checkpointing = False
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scene_graphs: List[Dict[str, Any]],
        token_type_ids: Optional[torch.Tensor] = None,
        question_types: Optional[List[str]] = None,
        image_widths: Optional[torch.Tensor] = None,
        image_heights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:  # Return type updated to Dict
        """
        Forward pass for the MIMIC-CXR VQA model.
        
        Returns:
            A dictionary containing model outputs.
        """
        device = images.device
        batch_size = images.shape[0]
        
        # CRITICAL FIX: Slice scene_graphs to match the local batch size on this GPU
        # DataParallel automatically splits tensor inputs but NOT list inputs
        # This ensures scene_graphs matches the local batch size after splitting
        local_scene_graphs = scene_graphs[:batch_size]
        
        # Store image dimensions for potential downstream use
        self._image_widths = image_widths
        self._image_heights = image_heights
        
        # Extract visual features
        # Get bboxes from scene graphs (using local_scene_graphs)
        bboxes = []
        for sg in local_scene_graphs:
            sg_bboxes = torch.tensor(sg['bboxes'], dtype=torch.float, device=device)
            bboxes.append(sg_bboxes)
        
        visual_features = self.visual_encoder(images, bboxes)  # (B, N, 512)
        visual_features = self.visual_proj(visual_features)  # (B, N, D)
        
        # Create visual mask (using local_scene_graphs)
        visual_mask = torch.zeros(batch_size, visual_features.shape[1], device=device)
        for i, sg in enumerate(local_scene_graphs):
            num_objects = min(sg['num_objects'], visual_features.shape[1])
            visual_mask[i, :num_objects] = 1.0
        
        # Encode text
        text_features, text_pooled = self.text_encoder(
            input_ids, attention_mask, token_type_ids
        )  # (B, L, D), (B, D)
        
        # Encode scene graphs (using local_scene_graphs)
        scene_features, scene_mask = self.scene_encoder(local_scene_graphs, device)  # (B, N, 134), (B, N)
        scene_features = self.scene_proj(scene_features)  # (B, N, D)
        
        # Scene-Embedded Interaction
        fused = self.sim(
            visual_features, text_features, scene_features,
            visual_mask=visual_mask,
            text_mask=attention_mask.float(),
            scene_mask=scene_mask
        )  # (B, D)
        
        # Multi-head VQA predictions
        vqa_logits = self.answer_module(fused)
        
        # CheXpert auxiliary prediction
        chexpert_logits = None
        if self.use_chexpert_head and self.chexpert_head is not None:
            # Use visual features for CheXpert (image-level labels)
            visual_pooled = (visual_features * visual_mask.unsqueeze(-1)).sum(1) / visual_mask.sum(1, keepdim=True).clamp(min=1)
            chexpert_logits = self.chexpert_head(visual_pooled)
        
        # Create the output object
        output = MIMICVQAOutput(
            vqa_logits=vqa_logits,
            chexpert_logits=chexpert_logits,
            pooled_output=fused
        )

        # Explicitly convert to a dictionary before returning
        return dict(output)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load model from checkpoint."""
        import json
        from pathlib import Path
        
        path = Path(path)
        
        # Load config
        config_path = path / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            # Extract model config
            model_config = config.get('model', config)
            kwargs.update({
                'hidden_size': model_config.get('hidden_size', 768),
                'num_hidden_layers': model_config.get('num_hidden_layers', 6),
                'num_attention_heads': model_config.get('num_attention_heads', 12),
            })
        
        # Create model
        model = cls(**kwargs)
        
        # Load weights
        weights_path = path / 'pytorch_model.bin'
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
        
        return model
    
    def save_pretrained(self, path: str):
        """Save model to checkpoint."""
        import json
        from pathlib import Path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        torch.save(self.state_dict(), path / 'pytorch_model.bin')
        
        # Save config
        config = {
            'hidden_size': self.hidden_size,
            'use_chexpert_head': self.use_chexpert_head,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
