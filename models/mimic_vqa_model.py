"""
MIMIC-CXR VQA Model - Complete Implementation with ALL Features

Single model with ALL capabilities built-in:
- ConvNeXt-Base visual backbone
- Bio+ClinicalBERT text encoder  
- Scene graph encoding
- Multi-head VQA classification
- Free-form answer generation (Decoder + Templates)
- CheXpert auxiliary classification
- Attention visualization for explainability
- Scene graph generation
- Visual grounding for answer localization

Usage:
======
    from models import MIMICCXRVQAModel
    
    model = MIMICCXRVQAModel()
    outputs = model(images, input_ids, attention_mask, scene_graphs)
    
    # Classification answers
    vqa_logits = outputs['vqa_logits']
    
    # Generated free-form answers
    generated_ids = outputs['generated_answer_ids']  # Token IDs
    generated_text = outputs['generated_answer_text']  # Decoded text (if tokenizer provided)
    template_answer = outputs['template_answer']  # Template-based answer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# =============================================================================
# MEDICAL VOCABULARY FOR ANSWER GENERATION
# =============================================================================

# Finding categories
FINDING_NAMES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# Anatomical regions
REGION_NAMES = [
    "right lung", "left lung", "right upper lobe", "right middle lobe", "right lower lobe",
    "left upper lobe", "left lower lobe", "mediastinum", "heart", "aorta",
    "trachea", "right hilum", "left hilum", "spine", "right costophrenic angle",
    "left costophrenic angle", "right hemidiaphragm", "left hemidiaphragm",
    "right chest wall", "left chest wall", "abdomen", "right clavicle",
    "left clavicle", "right shoulder", "left shoulder", "neck"
]

# Severity levels
SEVERITY_NAMES = ["mild", "moderate", "severe", "critical"]

# Answer templates
ANSWER_TEMPLATES = {
    'binary_yes': "Yes, {finding} is present in the {region}.",
    'binary_no': "No, there is no evidence of {finding}.",
    'finding': "The image shows {finding} in the {region}.",
    'location': "The abnormality is located in the {region}.",
    'severity': "The {finding} appears to be {severity} in nature.",
    'normal': "The chest X-ray appears normal with no significant abnormalities.",
    'multiple': "Multiple findings are present including {findings}.",
}


# =============================================================================
# OUTPUT CONTAINER
# =============================================================================

class MIMICVQAOutput(dict):
    """Output container that behaves like a dict and supports attribute access."""

    def __init__(self, vqa_logits, chexpert_logits, pooled_output, hidden_states=None):
        super().__init__(
            vqa_logits=vqa_logits,
            chexpert_logits=chexpert_logits,
            pooled_output=pooled_output,
            hidden_states=hidden_states,
        )
        for k, v in self.items():
            object.__setattr__(self, k, v)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value
        object.__setattr__(self, name, value)

    def __iter__(self):
        return iter(self.keys())


# =============================================================================
# VISUAL BACKBONE
# =============================================================================

class ConvNeXtFeatureExtractor(nn.Module):
    """ConvNeXt-Base visual backbone for chest X-ray images."""
    
    def __init__(self, model_name: str = 'convnext_base', pretrained: bool = True, output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim
        
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=[-1])
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feats = self.backbone(dummy)
                self.backbone_dim = feats[-1].shape[1]
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            )
            self.backbone_dim = 512
        
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.backbone_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )
        self.roi_pool_size = 7
    
    def forward(self, images: torch.Tensor, bboxes: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        # Cast images to match model dtype (handles FP16/mixed precision with DeepSpeed)
        try:
            first_param = next(self.backbone.parameters())
            if images.dtype != first_param.dtype:
                images = images.to(dtype=first_param.dtype)
        except StopIteration:
            pass  # No parameters found, skip casting
        
        if TIMM_AVAILABLE:
            feature_maps = self.backbone(images)[-1]
        else:
            feature_maps = self.backbone(images)
        
        if bboxes is None:
            return self.projection(feature_maps)
        
        batch_size, device = images.shape[0], images.device
        max_objects = max(len(b) for b in bboxes) if bboxes else 1
        all_features = []
        
        for b in range(batch_size):
            if b < len(bboxes) and len(bboxes[b]) > 0:
                img_bboxes = bboxes[b]
                roi_features = []
                
                for box in img_bboxes:
                    h, w = feature_maps.shape[2:]
                    x1, y1 = max(0, int(box[0] * w)), max(0, int(box[1] * h))
                    x2, y2 = min(w, int(box[2] * w)), min(h, int(box[3] * h))
                    if x2 <= x1: x2 = x1 + 1
                    if y2 <= y1: y2 = y1 + 1
                    
                    roi = feature_maps[b:b+1, :, y1:y2, x1:x2]
                    pooled = F.adaptive_avg_pool2d(roi, (1, 1))
                    roi_features.append(pooled)
                
                roi_features = torch.cat(roi_features, dim=0).flatten(1)
                roi_features = self.projection[2:](roi_features)
                
                if len(img_bboxes) < max_objects:
                    padding = torch.zeros(max_objects - len(img_bboxes), self.output_dim, device=device)
                    roi_features = torch.cat([roi_features, padding], dim=0)
                all_features.append(roi_features)
            else:
                global_feat = self.projection(feature_maps[b:b+1])
                padding = torch.zeros(max_objects - 1, self.output_dim, device=device)
                all_features.append(torch.cat([global_feat, padding], dim=0))
        
        return torch.stack(all_features)
    
    def get_feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        """Get raw feature maps for scene graph generation."""
        # Cast images to match model dtype (handles FP16/mixed precision with DeepSpeed)
        try:
            first_param = next(self.backbone.parameters())
            if images.dtype != first_param.dtype:
                images = images.to(dtype=first_param.dtype)
        except StopIteration:
            pass
        
        if TIMM_AVAILABLE:
            return self.backbone(images)[-1]
        return self.backbone(images)


# =============================================================================
# TEXT ENCODER
# =============================================================================

class TextEncoder(nn.Module):
    """Bio+ClinicalBERT text encoder for medical questions."""
    
    def __init__(self, model_name: str = 'emilyalsentzer/Bio_ClinicalBERT', output_dim: int = 768, freeze_layers: int = 0):
        super().__init__()
        self.output_dim = output_dim
        
        if TRANSFORMERS_AVAILABLE:
            self.encoder = AutoModel.from_pretrained(model_name)
            if freeze_layers > 0:
                for param in self.encoder.embeddings.parameters():
                    param.requires_grad = False
                for layer in self.encoder.encoder.layer[:freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            self.encoder = None
            self.embedding = nn.Embedding(30522, output_dim)
            self.lstm = nn.LSTM(output_dim, output_dim // 2, bidirectional=True, batch_first=True)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is not None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
            return outputs.last_hidden_state, outputs.pooler_output
        else:
            embeds = self.embedding(input_ids)
            output, (h, c) = self.lstm(embeds)
            pooled = torch.cat([h[-2], h[-1]], dim=1)
            return output, pooled


# =============================================================================
# SCENE GRAPH ENCODER
# =============================================================================

class SceneGraphEncoder(nn.Module):
    """Encodes scene graph information into 134-dim features."""
    
    def __init__(self, num_regions: int = 310, num_entities: int = 237, embedding_dim: int = 64):
        super().__init__()
        self.output_dim = 6 + embedding_dim * 2
        
        self.region_embedding = nn.Embedding(num_regions + 1, embedding_dim)
        self.entity_embedding = nn.Embedding(num_entities + 1, embedding_dim)
        self.region_proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())
        self.entity_proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())
    
    def forward(self, scene_graphs: List[Dict[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        max_objects = max(sg['num_objects'] for sg in scene_graphs) if scene_graphs else 1
        all_features, all_masks = [], []
        
        for sg in scene_graphs:
            num_objects = sg['num_objects']
            bboxes = torch.tensor(sg['bboxes'], dtype=torch.float, device=device)
            region_ids = torch.tensor(sg['region_ids'], dtype=torch.long, device=device)
            entity_ids = torch.tensor(sg['entity_ids'], dtype=torch.long, device=device)
            
            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            w, h = x2 - x1, y2 - y1
            bbox_features = torch.stack([x1, y1, x2, y2, w * h, w / (h + 1e-6)], dim=1)
            
            region_emb = self.region_proj(self.region_embedding(region_ids))
            entity_emb = self.entity_proj(self.entity_embedding(entity_ids))
            
            features = torch.cat([bbox_features, region_emb, entity_emb], dim=1)
            mask = torch.ones(num_objects, device=device)
            
            if num_objects < max_objects:
                features = torch.cat([features, torch.zeros(max_objects - num_objects, self.output_dim, device=device)], dim=0)
                mask = torch.cat([mask, torch.zeros(max_objects - num_objects, device=device)], dim=0)
            
            all_features.append(features)
            all_masks.append(mask)
        
        return torch.stack(all_features), torch.stack(all_masks)


# =============================================================================
# SCENE-EMBEDDED INTERACTION MODULE
# =============================================================================

class SceneEmbeddedInteraction(nn.Module):
    """Scene-Embedded Interaction Module with attention extraction for explainability."""
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.visual_to_text = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        self.text_to_visual = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        self.scene_to_text = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        
        self.visual_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.text_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.scene_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, visual_features, text_features, scene_features, visual_mask=None, text_mask=None, scene_mask=None):
        # Get authoritative dtype from parameters (handles DeepSpeed FP16)
        try:
            dt = next(self.ffn.parameters()).dtype
        except StopIteration:
            dt = visual_features.dtype
        
        visual_key_padding_mask = ~visual_mask.bool() if visual_mask is not None else None
        text_key_padding_mask = ~text_mask.bool() if text_mask is not None else None
        scene_key_padding_mask = ~scene_mask.bool() if scene_mask is not None else None
        
        all_text_to_visual_attn = []
        all_visual_to_text_attn = []
        all_scene_to_text_attn = []
        
        for i in range(self.num_layers):
            visual_attended, v2t_attn = self.visual_to_text[i](visual_features, text_features, text_features, key_padding_mask=text_key_padding_mask, need_weights=True, average_attn_weights=False)
            visual_features = self.visual_norms[i](visual_features + visual_attended)
            
            text_attended, t2v_attn = self.text_to_visual[i](text_features, visual_features, visual_features, key_padding_mask=visual_key_padding_mask, need_weights=True, average_attn_weights=False)
            text_features = self.text_norms[i](text_features + text_attended)
            
            scene_attended, s2t_attn = self.scene_to_text[i](scene_features, text_features, text_features, key_padding_mask=text_key_padding_mask, need_weights=True, average_attn_weights=False)
            scene_features = self.scene_norms[i](scene_features + scene_attended)
            
            all_text_to_visual_attn.append(t2v_attn)
            all_visual_to_text_attn.append(v2t_attn)
            all_scene_to_text_attn.append(s2t_attn)
        
        # Pool each modality (cast masks to dt to prevent fp32 upcast from broadcasting)
        if visual_mask is not None:
            vm = visual_mask.unsqueeze(-1).to(dtype=dt)
            visual_pooled = (visual_features * vm).sum(1) / vm.sum(1).clamp(min=1)
        else:
            visual_pooled = visual_features.mean(1)
        
        if text_mask is not None:
            tm = text_mask.unsqueeze(-1).to(dtype=dt)
            text_pooled = (text_features * tm).sum(1) / tm.sum(1).clamp(min=1)
        else:
            text_pooled = text_features.mean(1)
        
        if scene_mask is not None:
            sm = scene_mask.unsqueeze(-1).to(dtype=dt)
            scene_pooled = (scene_features * sm).sum(1) / sm.sum(1).clamp(min=1)
        else:
            scene_pooled = scene_features.mean(1)
        
        combined = visual_pooled + text_pooled + scene_pooled
        # Ensure dtype matches FFN params before Linear layers (DeepSpeed FP16 safety)
        if combined.dtype != dt:
            combined = combined.to(dtype=dt)
        output = self.ffn_norm(combined + self.ffn(combined))
        
        attention_dict = {
            'text_to_visual': torch.stack(all_text_to_visual_attn, dim=0).mean(0),
            'visual_to_text': torch.stack(all_visual_to_text_attn, dim=0).mean(0),
            'scene_to_text': torch.stack(all_scene_to_text_attn, dim=0).mean(0),
        }
        
        return output, attention_dict, text_features


# =============================================================================
# CLASSIFICATION ANSWER MODULE
# =============================================================================

class MultiHeadAnswerModule(nn.Module):
    """Multi-head classification answer module for different question types."""
    
    def __init__(self, hidden_size: int = 768, num_binary_classes: int = 2, num_category_classes: int = 14, num_region_classes: int = 26, num_severity_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.binary_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_binary_classes))
        self.category_head = nn.Sequential(nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_category_classes))
        self.region_head = nn.Sequential(nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_region_classes))
        self.severity_head = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_severity_classes))
    
    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Ensure input matches head dtype (DeepSpeed FP16 safety)
        try:
            dt = next(self.binary_head.parameters()).dtype
            if pooled_output.dtype != dt:
                pooled_output = pooled_output.to(dtype=dt)
        except StopIteration:
            pass
        return {'binary': self.binary_head(pooled_output), 'category': self.category_head(pooled_output), 'region': self.region_head(pooled_output), 'severity': self.severity_head(pooled_output)}


# =============================================================================
# FREE-FORM ANSWER DECODER (Transformer Decoder) - PRIMARY ANSWER GENERATOR
# =============================================================================

class AnswerDecoder(nn.Module):
    """
    PRIMARY ANSWER GENERATOR: Transformer decoder for generating report-quality answers.
    
    This decoder is trained on the rich, hierarchical answers from MIMIC-Ext-CXR-QBA:
    
    MIMIC-Ext-CXR-QBA Answer Features:
    - Full sentences derived from actual radiology report sentences
    - Hierarchical structure: main_answer + details + related_information
    - Generated via 4 strategies: Indication, Abnormal, Region, Finding
    - Rich tags: findings, regions, severity, certainty, modifiers, changes
    - Bounding boxes for answer localization
    
    Training:
    - Input: answer_ids from dataset (tokenized hierarchical answer text)
    - Target: Full answer text like "There is no focal consolidation. The lungs appear clear..."
    - Loss: Cross-entropy with teacher forcing
    
    Inference:
    - Generates report-style answers autoregressively
    - Output: generated_answer_text (decoded tokens)
    
    Takes the fused multimodal representation and generates natural language answers
    token by token using autoregressive decoding.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,  # BERT vocab size
        num_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 64,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 101,  # [CLS]
        eos_token_id: int = 102,  # [SEP]
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Memory projection (from fused features)
        self.memory_proj = nn.Linear(hidden_size, hidden_size)
    
    def _get_param_dtype(self) -> torch.dtype:
        """Get authoritative dtype from module parameters (handles DeepSpeed FP16)."""
        try:
            return next(self.memory_proj.parameters()).dtype
        except StopIteration:
            return torch.float32
    
    def forward(
        self,
        fused_features: torch.Tensor,      # (B, D) or (B, N, D)
        encoder_hidden: torch.Tensor,      # (B, S, D) encoder outputs for cross-attention
        target_ids: Optional[torch.Tensor] = None,  # (B, T) target token ids for teacher forcing
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for answer generation.
        
        Args:
            fused_features: Fused multimodal representation
            encoder_hidden: Encoder outputs for cross-attention
            target_ids: Target tokens for teacher forcing (training)
            encoder_mask: Mask for encoder outputs
            
        Returns:
            logits: (B, T, V) vocabulary logits
            generated_ids: (B, T) generated token ids (inference only)
        """
        device = fused_features.device
        batch_size = fused_features.shape[0]
        dt = self._get_param_dtype()
        
        # Ensure inputs match parameter dtype (DeepSpeed FP16 safety)
        if fused_features.dtype != dt:
            fused_features = fused_features.to(dtype=dt)
        if encoder_hidden.dtype != dt:
            encoder_hidden = encoder_hidden.to(dtype=dt)
        
        # Prepare memory for cross-attention
        if fused_features.dim() == 2:
            # Expand fused features to sequence
            memory = self.memory_proj(fused_features).unsqueeze(1)  # (B, 1, D)
            memory = torch.cat([memory, encoder_hidden], dim=1)  # (B, 1+S, D)
        else:
            memory = torch.cat([self.memory_proj(fused_features), encoder_hidden], dim=1)
        
        if target_ids is not None:
            # Teacher forcing (training)
            return self._forward_train(memory, target_ids, encoder_mask)
        else:
            # Autoregressive generation (inference)
            return self._forward_generate(memory, batch_size, device, encoder_mask)
    
    def _forward_train(self, memory: torch.Tensor, target_ids: torch.Tensor, encoder_mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training forward with teacher forcing."""
        B, T = target_ids.shape
        device = target_ids.device
        dt = memory.dtype  # Use memory dtype as reference
        
        # Embed target tokens
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        tgt_embeds = self.token_embedding(target_ids) + self.position_embedding(positions)
        tgt_embeds = self.dropout(self.layer_norm(tgt_embeds))
        
        # Ensure embeddings match memory dtype (LayerNorm may upcast)
        if tgt_embeds.dtype != dt:
            tgt_embeds = tgt_embeds.to(dtype=dt)
        
        # Causal mask - must match tensor dtype for some PyTorch versions
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        if tgt_mask.dtype != dt:
            tgt_mask = tgt_mask.to(dtype=dt)
        
        # Decode
        decoder_out = self.decoder(tgt_embeds, memory, tgt_mask=tgt_mask)
        # Guard after decoder (internal LayerNorm may upcast)
        if decoder_out.dtype != dt:
            decoder_out = decoder_out.to(dtype=dt)
        logits = self.output_proj(decoder_out)
        
        return {'logits': logits, 'generated_ids': target_ids}
    
    def _forward_generate(self, memory: torch.Tensor, batch_size: int, device: torch.device, encoder_mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Autoregressive generation."""
        dt = memory.dtype  # Use memory dtype as reference
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(self.max_length - 1):
            T = generated.shape[1]
            positions = torch.arange(T, device=device).unsqueeze(0).expand(batch_size, -1)
            tgt_embeds = self.token_embedding(generated) + self.position_embedding(positions)
            tgt_embeds = self.layer_norm(tgt_embeds)
            if tgt_embeds.dtype != dt:
                tgt_embeds = tgt_embeds.to(dtype=dt)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            if tgt_mask.dtype != dt:
                tgt_mask = tgt_mask.to(dtype=dt)
            
            decoder_out = self.decoder(tgt_embeds, memory, tgt_mask=tgt_mask)
            if decoder_out.dtype != dt:
                decoder_out = decoder_out.to(dtype=dt)
            logits = self.output_proj(decoder_out[:, -1:, :])  # Last token
            
            next_token = logits.argmax(dim=-1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == self.eos_token_id).all():
                break
        
        # Get final logits for the full sequence
        T = generated.shape[1]
        positions = torch.arange(T, device=device).unsqueeze(0).expand(batch_size, -1)
        tgt_embeds = self.token_embedding(generated) + self.position_embedding(positions)
        tgt_embeds = self.layer_norm(tgt_embeds)
        if tgt_embeds.dtype != dt:
            tgt_embeds = tgt_embeds.to(dtype=dt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        if tgt_mask.dtype != dt:
            tgt_mask = tgt_mask.to(dtype=dt)
        decoder_out = self.decoder(tgt_embeds, memory, tgt_mask=tgt_mask)
        if decoder_out.dtype != dt:
            decoder_out = decoder_out.to(dtype=dt)
        logits = self.output_proj(decoder_out)
        
        return {'logits': logits, 'generated_ids': generated}


# =============================================================================
# CLASSIFICATION-BASED ANSWER GENERATOR (Fallback/Legacy)
# =============================================================================

class TemplateAnswerGenerator(nn.Module):
    """
    FALLBACK/LEGACY: Simple template-based answer generator from classification outputs.
    
    NOTE: This is NOT the primary answer generator! The model's transformer decoder
    (AnswerDecoder) is trained on the rich, hierarchical, report-derived answers from
    MIMIC-Ext-CXR-QBA dataset and should be used for production inference.
    
    This template generator is provided only as:
    - A fallback when decoder is not yet trained
    - A simple deterministic baseline for comparison
    - Quick interpretable outputs based on classification heads
    
    For high-quality answers, use the decoder's generated_answer_text output instead.
    
    The MIMIC-Ext-CXR-QBA dataset provides:
    - Hierarchical, multi-granular answers (main + details + related_info)
    - Full sentences derived from actual radiology reports
    - Rich tags (findings, regions, severity, certainty, modifiers)
    - Bounding boxes for answer localization
    
    The decoder is trained on these actual answers, not simple templates.
    """
    
    def __init__(self):
        super().__init__()
        self.finding_names = FINDING_NAMES
        self.region_names = REGION_NAMES
        self.severity_names = SEVERITY_NAMES
        self.templates = ANSWER_TEMPLATES
    
    def forward(
        self,
        vqa_logits: Dict[str, torch.Tensor],
        question_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate simple template-based answers from classification logits.
        
        WARNING: These are basic templates, NOT report-quality answers.
        Use the decoder output (generated_answer_text) for production.
        
        Args:
            vqa_logits: Dict with 'binary', 'category', 'region', 'severity' logits
            question_types: Optional list of question types for template selection
            
        Returns:
            List of simple answer strings (for fallback/baseline only)
        """
        batch_size = vqa_logits['binary'].shape[0]
        answers = []
        
        # Get predictions
        binary_preds = vqa_logits['binary'].argmax(dim=-1)
        category_preds = vqa_logits['category'].argmax(dim=-1)
        region_preds = vqa_logits['region'].argmax(dim=-1)
        severity_preds = vqa_logits['severity'].argmax(dim=-1)
        
        for b in range(batch_size):
            binary = binary_preds[b].item()
            category = category_preds[b].item()
            region = region_preds[b].item()
            severity = severity_preds[b].item()
            
            # Get names (with bounds checking)
            finding = self.finding_names[category] if category < len(self.finding_names) else "abnormality"
            location = self.region_names[region] if region < len(self.region_names) else "the chest"
            sev_level = self.severity_names[severity] if severity < len(self.severity_names) else "moderate"
            
            # Determine question type if provided
            q_type = question_types[b].lower() if question_types and b < len(question_types) else ""
            
            # Generate appropriate answer (simple templates)
            if category == 0:  # No Finding
                answer = self.templates['normal']
            elif 'where' in q_type or 'location' in q_type:
                answer = self.templates['location'].format(region=location)
            elif 'severe' in q_type or 'severity' in q_type:
                answer = self.templates['severity'].format(finding=finding, severity=sev_level)
            elif binary == 1:  # Yes
                answer = self.templates['binary_yes'].format(finding=finding, region=location)
            elif binary == 0:  # No
                answer = self.templates['binary_no'].format(finding=finding)
            else:
                answer = self.templates['finding'].format(finding=finding, region=location)
            
            # Add severity if relevant
            if category != 0 and severity > 0 and 'severe' not in q_type:
                answer = answer.rstrip('.') + f", appearing {sev_level} in severity."
            
            answers.append(answer)
        
        return answers


# =============================================================================
# CHEXPERT HEAD
# =============================================================================

class CheXpertHead(nn.Module):
    """Auxiliary classification head for CheXpert labels."""
    
    def __init__(self, hidden_size: int = 768, num_classes: int = 14, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_classes))
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        # Ensure input matches classifier dtype (DeepSpeed FP16 safety)
        try:
            dt = next(self.classifier.parameters()).dtype
            if pooled_output.dtype != dt:
                pooled_output = pooled_output.to(dtype=dt)
        except StopIteration:
            pass
        return self.classifier(pooled_output)


# =============================================================================
# SCENE GRAPH GENERATOR
# =============================================================================

class SceneGraphGenerator(nn.Module):
    """Scene Graph Generation Module - generates scene graphs from images."""
    
    def __init__(self, visual_dim: int = 1024, hidden_size: int = 768, num_entity_classes: int = 237, num_region_classes: int = 310, num_relationships: int = 10, max_objects: int = 20, dropout: float = 0.1):
        super().__init__()
        self.max_objects = max_objects
        self.roi_pool_size = 7
        
        # RPN
        self.rpn_conv = nn.Sequential(nn.Conv2d(visual_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.rpn_cls = nn.Conv2d(256, num_entity_classes + 1, 1)
        self.rpn_reg = nn.Conv2d(256, 4, 1)
        self.rpn_centerness = nn.Conv2d(256, 1, 1)
        
        # Classifiers
        roi_feat_dim = visual_dim * self.roi_pool_size * self.roi_pool_size
        self.entity_classifier = nn.Sequential(nn.Linear(roi_feat_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_entity_classes))
        self.region_classifier = nn.Sequential(nn.Linear(roi_feat_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_region_classes))
        self.positiveness_classifier = nn.Sequential(nn.Linear(roi_feat_dim, 256), nn.ReLU(), nn.Linear(256, 2))
        
        # Relationship predictor
        self.obj_proj = nn.Linear(roi_feat_dim, hidden_size)
        self.rel_classifier = nn.Sequential(nn.Linear(hidden_size * 2 + hidden_size // 4, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_relationships))
        self.spatial_encoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, hidden_size // 4))
    
    def _get_param_dtype(self) -> torch.dtype:
        """Get the dtype of this module's parameters (handles DeepSpeed FP16)."""
        try:
            return next(self.entity_classifier.parameters()).dtype
        except StopIteration:
            return torch.float32
    
    def forward(self, visual_features: torch.Tensor, gt_bboxes=None, gt_entities=None, gt_regions=None) -> Dict[str, Any]:
        B, C, H, W = visual_features.shape
        device = visual_features.device
        
        # Get the dtype that classifiers expect (DeepSpeed may convert params to fp16)
        # This is the authoritative dtype - all tensors must match before hitting Linear layers
        param_dtype = self._get_param_dtype()
        
        # Cast input to match parameter dtype for RPN conv layers
        if visual_features.dtype != param_dtype:
            visual_features = visual_features.to(dtype=param_dtype)
        
        rpn_feat = self.rpn_conv(visual_features)
        
        # BatchNorm2d may upcast to fp32 for numerical stability - cast back
        if rpn_feat.dtype != param_dtype:
            rpn_feat = rpn_feat.to(dtype=param_dtype)
        
        rpn_cls = self.rpn_cls(rpn_feat)
        rpn_reg = self.rpn_reg(rpn_feat)
        centerness = self.rpn_centerness(rpn_feat)
        
        # Get proposals
        centerness_flat = centerness.view(B, -1)
        scores, indices = centerness_flat.topk(min(self.max_objects, centerness_flat.shape[1]), dim=1)
        scores = torch.sigmoid(scores)
        
        bbox_flat = rpn_reg.view(B, 4, -1).permute(0, 2, 1)
        boxes = torch.zeros(B, self.max_objects, 4, dtype=param_dtype, device=device)
        for b in range(B):
            selected = bbox_flat[b, indices[b] % bbox_flat.shape[1]]
            boxes[b, :selected.shape[0]] = torch.sigmoid(selected)
        
        # ROI features - use param_dtype to match classifier weights
        roi_features = torch.zeros(B, self.max_objects, C * self.roi_pool_size * self.roi_pool_size, dtype=param_dtype, device=device)
        for b in range(B):
            for n in range(self.max_objects):
                box = boxes[b, n]
                # Clamp coordinates to valid feature-map bounds so the
                # crop always has ≥1 pixel in both H and W dimensions.
                x1 = max(0, min(int(box[0].item() * W), W - 1))
                y1 = max(0, min(int(box[1].item() * H), H - 1))
                x2 = max(x1 + 1, min(int(box[2].item() * W) + 1, W))
                y2 = max(y1 + 1, min(int(box[3].item() * H) + 1, H))
                roi = visual_features[b:b+1, :, y1:y2, x1:x2]
                pooled = F.adaptive_avg_pool2d(roi, (self.roi_pool_size, self.roi_pool_size))
                roi_features[b, n] = pooled.flatten().to(dtype=param_dtype)
        
        entity_logits = self.entity_classifier(roi_features)
        region_logits = self.region_classifier(roi_features)
        positiveness_logits = self.positiveness_classifier(roi_features)
        
        # Relationships
        obj_features = self.obj_proj(roi_features)
        N = self.max_objects
        subj_exp = obj_features.unsqueeze(2).expand(B, N, N, -1)
        obj_exp = obj_features.unsqueeze(1).expand(B, N, N, -1)
        
        subj_bbox = boxes.unsqueeze(2).expand(B, N, N, 4)
        obj_bbox = boxes.unsqueeze(1).expand(B, N, N, 4)
        spatial_input = torch.cat([subj_bbox, obj_bbox], dim=-1).to(dtype=param_dtype)
        spatial = self.spatial_encoder(spatial_input)
        
        rel_input = torch.cat([subj_exp, obj_exp, spatial], dim=-1)
        relationship_logits = self.rel_classifier(rel_input)
        
        return {
            'bbox_preds': boxes, 'entity_logits': entity_logits, 'region_logits': region_logits,
            'positiveness_logits': positiveness_logits, 'relationship_logits': relationship_logits,
            'objectness_scores': scores, 'rpn_cls_logits': rpn_cls, 'rpn_bbox_preds': rpn_reg, 'rpn_centerness': centerness
        }


# =============================================================================
# VISUAL GROUNDING HEAD (Scene Graph-Guided)
# =============================================================================

class VisualGroundingHead(nn.Module):
    """
    Visual Grounding Head for answer localization.
    
    Integrates scene graph information for better localization:
    - Uses scene graph features to guide attention
    - Combines visual and scene graph cues for bbox prediction
    """
    
    def __init__(self, hidden_size: int = 768, num_visual_features: int = 49, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Query projection from fused features
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.visual_proj = nn.Linear(hidden_size, hidden_size)
        
        # Scene graph-guided attention
        self.scene_guide_proj = nn.Linear(hidden_size, hidden_size)
        self.grounding_attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        
        # Scene graph integration gate
        self.scene_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Bbox and pointing heads
        self.bbox_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 4))
        self.pointing_head = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
    
    def forward(
        self, 
        fused_features: torch.Tensor, 
        visual_features: torch.Tensor, 
        visual_mask: Optional[torch.Tensor] = None,
        scene_features: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: (B, D) multimodal fused representation
            visual_features: (B, N_vis, D) visual region features
            visual_mask: (B, N_vis) mask for visual features
            scene_features: (B, N_sg, D) scene graph node features (optional)
            scene_mask: (B, N_sg) mask for scene graph (optional)
        """
        # Get authoritative dtype from parameters (handles DeepSpeed FP16)
        try:
            dt = next(self.query_proj.parameters()).dtype
        except StopIteration:
            dt = fused_features.dtype
        
        # Ensure inputs match parameter dtype
        if fused_features.dtype != dt:
            fused_features = fused_features.to(dtype=dt)
        if visual_features.dtype != dt:
            visual_features = visual_features.to(dtype=dt)
        
        query = self.query_proj(fused_features).unsqueeze(1)
        visual = self.visual_proj(visual_features)
        key_padding_mask = ~visual_mask.bool() if visual_mask is not None else None
        
        # Visual grounding attention
        grounding_out, spatial_attn = self.grounding_attention(
            query, visual, visual, 
            key_padding_mask=key_padding_mask, 
            need_weights=True
        )
        grounding_features = grounding_out.squeeze(1)
        # Guard: MultiheadAttention may produce fp32 output
        if grounding_features.dtype != dt:
            grounding_features = grounding_features.to(dtype=dt)
        
        # Scene graph-guided refinement (if scene features available)
        if scene_features is not None and scene_features.shape[1] > 0:
            if scene_features.dtype != dt:
                scene_features = scene_features.to(dtype=dt)
            # Pool scene graph features (cast mask to dt to prevent fp32 upcast)
            if scene_mask is not None:
                scene_mask_dt = scene_mask.unsqueeze(-1).to(dtype=dt)
                scene_pooled = (scene_features * scene_mask_dt).sum(1) / scene_mask_dt.sum(1).clamp(min=1)
            else:
                scene_pooled = scene_features.mean(1)
            
            # Gated fusion with scene graph
            if scene_pooled.dtype != dt:
                scene_pooled = scene_pooled.to(dtype=dt)
            scene_guide = self.scene_guide_proj(scene_pooled)
            gate = self.scene_gate(torch.cat([grounding_features, scene_guide], dim=-1))
            grounding_features = grounding_features + gate * scene_guide
        
        # Ensure dtype before final heads
        if grounding_features.dtype != dt:
            grounding_features = grounding_features.to(dtype=dt)
        
        bbox_pred = torch.sigmoid(self.bbox_head(grounding_features))
        pointing_score = self.pointing_head(grounding_features)
        
        return {
            'bbox_pred': bbox_pred, 
            'pointing_score': pointing_score, 
            'spatial_attention': spatial_attn.squeeze(1), 
            'grounding_features': grounding_features
        }


# =============================================================================
# MANIFOLD-CONSTRAINED HYPER-CONNECTIONS (mHC)
# Based on: "mHC: Manifold-Constrained Hyper-Connections" (Xie et al., 2024)
#
# Key contributions from paper:
# - Extends Hyper-Connections (HC) with manifold constraints
# - Restores identity mapping property lost in plain HC (Eq. 2 vs 4)
# - Uses Birkhoff polytope (doubly stochastic) via Sinkhorn-Knopp (Eq. 6-9)
# - Multiple diversified residual paths with learnable importance
# - RMSNorm + tanh for dynamic mappings (Eq. 5, 7-8)
#
# Manifold choices:
# - 'birkhoff': Doubly stochastic matrices via Sinkhorn-Knopp (PAPER DEFAULT)
# - 'sphere': Normalizes to unit sphere (simpler alternative)
# - 'oblique': Column-wise normalization (good for sparse features)
# - 'grassmann': Low-rank subspace projection (expensive but expressive)
# - 'stiefel': Orthonormal projection (alternative to Grassmann)
# =============================================================================


class RMSNorm(nn.Module):
    """RMSNorm as used in the mHC paper (Eq. 5)."""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def sinkhorn_knopp(x: torch.Tensor, t_max: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for projecting onto Birkhoff polytope.
    
    Projects a matrix onto the set of doubly stochastic matrices
    (non-negative, rows and columns sum to 1).
    
    From paper Section 4.2, Eq. 8-9:
    - First apply exp() for positivity
    - Then iterate: normalize rows, normalize columns
    
    Args:
        x: (*, D, D) or (*, D) input tensor
        t_max: Number of iterations (paper uses 20)
        eps: Small constant for numerical stability
        
    Returns:
        Doubly stochastic matrix/vector
    """
    # Apply exp for positivity (Eq. 8)
    x_pos = torch.exp(x - x.max(dim=-1, keepdim=True)[0])  # Subtract max for stability
    
    # Sinkhorn iterations (Eq. 9)
    for _ in range(t_max):
        # Row normalization
        x_pos = x_pos / (x_pos.sum(dim=-1, keepdim=True) + eps)
        # Column normalization (if 2D)
        if x_pos.dim() >= 2:
            x_pos = x_pos / (x_pos.sum(dim=-2, keepdim=True) + eps)
    
    return x_pos


class ManifoldProjection(nn.Module):
    """
    Projects residual connections onto a specific manifold to restore identity mapping.
    
    Implements manifold constraint: x' = x + α * P_M(f(x))
    where P_M is the manifold projection operator.
    
    Paper's primary manifold is Birkhoff polytope via Sinkhorn-Knopp (Eq. 6-9),
    which ensures:
    - Non-negativity (prevents signal cancellation)
    - Rows/cols sum to 1 (preserves mean, bounded spectral norm)
    - Compositional closure (stable under multiplication)
    
    Args:
        dim: Feature dimension
        manifold_type: 'birkhoff' (paper default), 'sphere', 'oblique', 'grassmann', 'stiefel'
        use_qr: Use QR decomposition instead of SVD for Grassmann (faster)
        sinkhorn_iters: Number of Sinkhorn iterations for Birkhoff (paper: 20)
    """
    
    def __init__(
        self, 
        dim: int, 
        manifold_type: str = 'birkhoff',  # Paper default
        use_qr: bool = True,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.dim = dim
        self.manifold_type = manifold_type
        self.use_qr = use_qr
        self.sinkhorn_iters = sinkhorn_iters
        
        # Learnable scaling factor (initialized small for stability, paper: 0.01)
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)
        
        # Manifold-specific parameters
        if manifold_type == 'sphere':
            self.radius = nn.Parameter(torch.ones(1))
        elif manifold_type in ('grassmann', 'stiefel'):
            self.rank = max(dim // 4, 1)
        elif manifold_type == 'birkhoff':
            # For Birkhoff, we need a square matrix transformation
            # Paper uses Hres as (dim, dim) matrix
            self.hres_weight = nn.Parameter(torch.randn(dim, dim) * 0.01)
    
    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Project residual onto manifold.
        
        Args:
            residual: (*, D) tensor to project
            
        Returns:
            Projected tensor with same shape
        """
        if self.manifold_type == 'birkhoff':
            # Birkhoff polytope via Sinkhorn-Knopp (Paper's main contribution)
            projected = self._birkhoff_project(residual)
            
        elif self.manifold_type == 'sphere':
            norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            projected = self.radius * residual / norm
            
        elif self.manifold_type == 'oblique':
            norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            projected = residual / norm
            
        elif self.manifold_type == 'grassmann':
            projected = self._grassmann_project(residual)
            
        elif self.manifold_type == 'stiefel':
            projected = self._stiefel_project(residual)
            
        else:
            projected = residual
        
        return self.alpha * projected
    
    def _birkhoff_project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project onto Birkhoff polytope (doubly stochastic matrices).
        
        From paper Eq. 6-9:
        - Hres is projected to doubly stochastic via Sinkhorn
        - Applied as: x @ Sinkhorn(exp(Hres))
        """
        # Get doubly stochastic matrix from weights
        ds_matrix = sinkhorn_knopp(self.hres_weight, self.sinkhorn_iters)
        
        # Apply transformation: x @ doubly_stochastic_matrix
        # This preserves mean and bounds spectral norm to 1
        projected = F.linear(x, ds_matrix)
        
        return projected
    
    def _grassmann_project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto Grassmann manifold (low-rank subspace)."""
        orig_shape = x.shape
        
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x
        
        if x_2d.shape[0] < self.rank:
            return x
        
        try:
            if self.use_qr:
                # QR decomposition (faster than SVD)
                Q, R = torch.linalg.qr(x_2d.T)  # (D, min(D, B))
                # Project: X @ Q @ Q.T
                projected = x_2d @ Q[:, :self.rank] @ Q[:, :self.rank].T
            else:
                # SVD decomposition (more accurate)
                U, S, Vh = torch.linalg.svd(x_2d, full_matrices=False)
                # Low-rank reconstruction
                projected = U[:, :self.rank] @ torch.diag(S[:self.rank]) @ Vh[:self.rank, :]
            
            return projected.reshape(orig_shape)
        except:
            return x
    
    def _stiefel_project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto Stiefel manifold (orthonormal columns)."""
        orig_shape = x.shape
        
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x
        
        try:
            # Polar decomposition: X = U @ P where U is orthonormal
            # Approximate via SVD: U = U_svd @ Vh
            U, S, Vh = torch.linalg.svd(x_2d, full_matrices=False)
            projected = U @ Vh
            return projected.reshape(orig_shape)
        except:
            return x


class HyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection block (Paper Eq. 5-9).
    
    Implements: y = x + α * Σ(w_i * P_M(f_i(residual)))
    
    Paper-faithful features:
    - Multiple residual pathways (expansion factor n, paper uses n=4)
    - RMSNorm + tanh for dynamic mappings (Eq. 5, 7)
    - Manifold projection (default: Birkhoff via Sinkhorn) (Eq. 6-9)
    - Learnable gating α initialized small (0.01) (Eq. 5)
    - Tracks Amax Gain Magnitude for stability analysis (Fig. 3, 7-8)
    
    Args:
        dim: Feature dimension
        num_paths: Number of hyper-connection paths (paper: n=4)
        manifold_type: 'birkhoff' (paper default), 'sphere', 'oblique', 'grassmann', 'stiefel'
        dropout: Dropout rate for path transforms
        use_qr: Use QR instead of SVD for Grassmann (faster)
        sinkhorn_iters: Iterations for Birkhoff projection (paper: 20)
    """
    
    def __init__(
        self, 
        dim: int, 
        num_paths: int = 4,  # Paper uses n=4
        manifold_type: str = 'birkhoff',  # Paper default
        dropout: float = 0.1,
        use_qr: bool = True,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.dim = dim
        self.num_paths = num_paths
        self.manifold_type = manifold_type
        
        # Path weights with sigmoid gating (Paper Eq. 5: α initialized small)
        self.path_weights = nn.Parameter(torch.ones(num_paths) / num_paths)
        
        # Manifold projection for each path
        self.manifold_projs = nn.ModuleList([
            ManifoldProjection(dim, manifold_type, use_qr=use_qr, sinkhorn_iters=sinkhorn_iters) 
            for _ in range(num_paths)
        ])
        
        # Paper-faithful path transformations: RMSNorm + Linear + tanh (Eq. 5, 7)
        # Dynamic mapping: tanh(RMSNorm(x) @ W_dynamic)
        # Static mapping: bias term
        self.path_rms_norms = nn.ModuleList([RMSNorm(dim) for _ in range(num_paths)])
        self.path_dynamic_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_paths)])
        self.path_static_biases = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(num_paths)])
        self.path_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_paths)])
        
        # Final gating: sigmoid with learnable scale (Paper Eq. 5: α)
        # Initialized small for stability
        self.gate_alpha = nn.Parameter(torch.ones(1) * 0.01)
        self.gate_proj = nn.Linear(dim * 2, dim)
        
        # Metrics tracking (Paper Fig. 3, 7-8: Amax Gain Magnitude)
        self._last_gate_values = None
        self._last_path_weights = None
        self._last_amax_gain = None  # For stability analysis
        self._input_amax = None
        self._output_amax = None
    
    def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        """
        Apply manifold-constrained hyper-connection.
        
        Args:
            x: Original input (for identity mapping)
            f_x: Transformed features (from sublayer)
            
        Returns:
            y = x + α * gate * weighted sum of manifold-projected residuals
        """
        # Ensure inputs match parameter dtype (DeepSpeed FP16 safety)
        try:
            dt = next(self.gate_proj.parameters()).dtype
            if x.dtype != dt:
                x = x.to(dtype=dt)
            if f_x.dtype != dt:
                f_x = f_x.to(dtype=dt)
        except StopIteration:
            pass
        
        residual = f_x - x
        
        # Track input Amax for stability analysis
        self._input_amax = x.abs().max().detach()
        
        # Normalize path weights via softmax
        weights = F.softmax(self.path_weights, dim=0)
        self._last_path_weights = weights.detach()
        
        # Apply each hyper-connection path with paper-faithful transforms
        path_outputs = []
        for i, manifold_proj in enumerate(self.manifold_projs):
            # Paper Eq. 5, 7: Dynamic mapping with RMSNorm + tanh
            # h_dynamic = tanh(RMSNorm(residual) @ W)
            normed = self.path_rms_norms[i](residual)
            dynamic = torch.tanh(self.path_dynamic_projs[i](normed))
            
            # Add static bias
            path_residual = dynamic + self.path_static_biases[i]
            path_residual = self.path_dropouts[i](path_residual)
            
            # Project onto manifold (Paper Eq. 6-9)
            projected = manifold_proj(path_residual)
            path_outputs.append(weights[i] * projected)
        
        # Sum all paths
        combined_residual = sum(path_outputs)
        
        # Gated addition with small alpha (Paper Eq. 5)
        gate_input = torch.cat([x, combined_residual], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        self._last_gate_values = gate.mean().detach()
        
        # Final output: identity + α * gate * manifold-projected residuals
        output = x + self.gate_alpha * gate * combined_residual
        
        # Track Amax Gain for stability analysis (Paper Fig. 3, 7-8)
        self._output_amax = output.abs().max().detach()
        if self._input_amax is not None and self._input_amax > 0:
            self._last_amax_gain = (self._output_amax / self._input_amax).item()
        
        return output
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get mHC-specific metrics for logging.
        
        Includes Amax Gain Magnitude for stability analysis (Paper Fig. 3, 7-8).
        """
        metrics = {}
        if self._last_path_weights is not None:
            for i, w in enumerate(self._last_path_weights):
                metrics[f'path_{i}_weight'] = w.item()
        if self._last_gate_values is not None:
            metrics['gate_mean'] = self._last_gate_values.item()
        if self._last_amax_gain is not None:
            metrics['amax_gain'] = self._last_amax_gain  # Key stability metric
        if self._input_amax is not None:
            metrics['input_amax'] = self._input_amax.item()
        if self._output_amax is not None:
            metrics['output_amax'] = self._output_amax.item()
        return metrics


class mHCBlock(nn.Module):
    """
    Complete mHC-enhanced transformer block (Paper macro-design, Section 2.2).
    
    Replaces standard residual connections with manifold-constrained hyper-connections.
    Designed to work with both sequence data and pooled features.
    
    Paper-faithful features:
    - Uses Birkhoff polytope by default (doubly stochastic via Sinkhorn)
    - n=4 expansion factor for HC paths
    - RMSNorm + tanh transforms
    - Tracks Amax Gain for stability monitoring
    
    Args:
        hidden_size: Feature dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        num_hc_paths: Number of hyper-connection paths (paper: n=4)
        manifold_type: 'birkhoff' (paper default), 'sphere', etc.
        dropout: Dropout rate
        use_qr: Use QR decomposition for Grassmann
        min_seq_len: Minimum sequence length for attention
        sinkhorn_iters: Iterations for Birkhoff projection (paper: 20)
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        ff_dim: int = 3072,
        num_hc_paths: int = 4,  # Paper uses n=4
        manifold_type: str = 'birkhoff',  # Paper default
        dropout: float = 0.1,
        use_qr: bool = True,
        min_seq_len: int = 4,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.min_seq_len = min_seq_len
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
        
        # mHC for both attention and FF residuals (paper: n=4 paths, Birkhoff default)
        self.attn_mhc = HyperConnection(
            hidden_size, num_hc_paths, manifold_type, dropout, use_qr, sinkhorn_iters
        )
        self.ff_mhc = HyperConnection(
            hidden_size, num_hc_paths, manifold_type, dropout, use_qr, sinkhorn_iters
        )
        
        # Learnable positional embeddings for short sequences
        self.pos_embed = nn.Parameter(torch.randn(1, min_seq_len, hidden_size) * 0.02)
        
        # For gradient checkpointing
        self._gradient_checkpointing = False
    
    def _get_param_dtype(self) -> torch.dtype:
        """Get authoritative dtype from module parameters (handles DeepSpeed FP16)."""
        try:
            return next(self.ff.parameters()).dtype
        except StopIteration:
            return torch.float32
    
    def _attention_block(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Attention sub-block (for gradient checkpointing)."""
        dt = self._get_param_dtype()
        if x.dtype != dt:
            x = x.to(dtype=dt)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        if attn_out.dtype != dt:
            attn_out = attn_out.to(dtype=dt)
        return self.attn_mhc(x, self.attn_norm(x + attn_out))
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward sub-block (for gradient checkpointing)."""
        dt = self._get_param_dtype()
        if x.dtype != dt:
            x = x.to(dtype=dt)
        ff_out = self.ff(x)
        return self.ff_mhc(x, self.ff_norm(x + ff_out))
    
    def forward(
        self, 
        x: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with mHC-enhanced residuals.
        
        Args:
            x: (B, L, D) input features
            key_padding_mask: (B, L) padding mask
            
        Returns:
            (B, L, D) transformed features
        """
        B, L, D = x.shape
        original_len = L
        
        # Handle short sequences by padding to min_seq_len
        if L < self.min_seq_len:
            pad_len = self.min_seq_len - L
            x = F.pad(x, (0, 0, 0, pad_len), value=0)  # Pad sequence dim
            # Add positional info to padded positions
            x[:, original_len:, :] = x[:, original_len:, :] + self.pos_embed[:, :pad_len, :]
            
            # Update mask
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, pad_len), value=True)  # Mask padded
            else:
                key_padding_mask = torch.zeros(B, self.min_seq_len, dtype=torch.bool, device=x.device)
                key_padding_mask[:, original_len:] = True
        
        # Self-attention with mHC
        if self._gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._attention_block, x, key_padding_mask, use_reentrant=False
            )
        else:
            x = self._attention_block(x, key_padding_mask)
        
        # Feed-forward with mHC
        if self._gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._ff_block, x, use_reentrant=False
            )
        else:
            x = self._ff_block(x)
        
        # Remove padding if added
        if original_len < self.min_seq_len:
            x = x[:, :original_len, :]
        
        return x
    
    def get_metrics(self) -> Dict[str, float]:
        """Get mHC-specific metrics from both attention and FF sub-blocks."""
        metrics = {}
        attn_metrics = self.attn_mhc.get_metrics()
        for k, v in attn_metrics.items():
            metrics[f'attn_mhc_{k}'] = v
        ff_metrics = self.ff_mhc.get_metrics()
        for k, v in ff_metrics.items():
            metrics[f'ff_mhc_{k}'] = v
        return metrics


# =============================================================================
# MAIN MODEL CLASS - ALL FEATURES INCLUDED
# =============================================================================

class MIMICCXRVQAModel(nn.Module):
    """
    Complete SSG-VQA-Net for MIMIC-CXR VQA with ALL features.
    
    ALL capabilities built-in:
    - Classification-based VQA (multi-head for binary/category/region/severity)
    - FREE-FORM ANSWER GENERATION (PRIMARY): Transformer decoder trained on
      MIMIC-Ext-CXR-QBA's rich, hierarchical, report-derived answers
    - Template-based answers (FALLBACK): Simple deterministic outputs from classification
    - Attention visualization for explainability
    - Scene graph generation from images
    - Visual grounding for answer localization (scene graph-guided)
    - CheXpert auxiliary classification
    - Manifold-Constrained Hyper-Connections (mHC) for enhanced feature fusion
    
    Answer Generation Strategy:
    ==========================
    The MIMIC-Ext-CXR-QBA dataset provides 42M+ QA pairs with:
    - Full sentence answers derived from actual radiology reports
    - Hierarchical structure: main_answer + details + related_information
    - 4 generation strategies: Indication, Abnormal, Region, Finding
    - Rich metadata: findings, regions, severity, certainty, bounding boxes
    
    The decoder (AnswerDecoder) is trained on these actual answers via:
    - Input: 'answer_ids' (tokenized full_answer_text from dataset)
    - Output: 'generated_answer_text' (model-generated report-style answer)
    
    The template generator (TemplateAnswerGenerator) is ONLY a fallback for:
    - Testing before decoder is trained
    - Quick deterministic outputs from classification heads
    
    Training Modes:
    - 'standard': Focus on VQA (SG generator frozen)
    - 'pretrain': All features emphasized (everything trainable)
    - 'finetune': Freeze scene graph gen, focus on VQA + generation + grounding
    
    Flexible Inference:
    - Works with partial features (only image + question required)
    - Scene graphs, grounding targets, answer_ids are all optional at inference
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
        vocab_size: int = 30522,
        max_answer_length: int = 64,
        decoder_layers: int = 4,
        dropout: float = 0.1,
        use_chexpert_head: bool = True,
        use_mhc: bool = True,  # Enable Manifold-Constrained Hyper-Connections
        mhc_manifold: str = 'birkhoff',  # Paper default: 'birkhoff', alternatives: 'sphere', 'oblique', 'grassmann', 'stiefel'
        num_mhc_paths: int = 4,  # Paper uses n=4
        sinkhorn_iters: int = 20,  # Paper uses 20 iterations for Birkhoff
        gradient_checkpointing: bool = False,
        training_mode: str = 'standard',
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        self.training_mode = training_mode
        self.use_chexpert_head = use_chexpert_head
        self.use_mhc = use_mhc
        self._visual_feature_dim = visual_feature_dim
        self.vocab_size = vocab_size
        
        # Visual backbone
        self.visual_encoder = ConvNeXtFeatureExtractor(model_name=visual_backbone, pretrained=True, output_dim=visual_feature_dim)
        
        # Text encoder
        self.text_encoder = TextEncoder(model_name=text_encoder, output_dim=hidden_size)
        
        # Scene graph encoder
        self.scene_encoder = SceneGraphEncoder(num_regions=num_regions, num_entities=num_entities, embedding_dim=64)
        
        # Feature projections
        self.visual_proj = nn.Linear(visual_feature_dim, hidden_size)
        self.scene_proj = nn.Linear(scene_graph_dim, hidden_size)
        
        # Scene-Embedded Interaction (with attention extraction)
        self.sim = SceneEmbeddedInteraction(hidden_size=hidden_size, num_heads=num_attention_heads, num_layers=sim_layers, dropout=dropout)
        
        # Manifold-Constrained Hyper-Connection for enhanced fusion
        # Paper: Uses Birkhoff polytope (doubly stochastic) via Sinkhorn-Knopp, n=4 paths
        if use_mhc:
            self.mhc_fusion = mHCBlock(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                ff_dim=hidden_size * 4,
                num_hc_paths=num_mhc_paths,
                manifold_type=mhc_manifold,
                dropout=dropout,
                sinkhorn_iters=sinkhorn_iters,
            )
        else:
            self.mhc_fusion = None
        
        # Classification answer module
        self.answer_module = MultiHeadAnswerModule(hidden_size=hidden_size, num_binary_classes=num_binary_classes, num_category_classes=num_category_classes, num_region_classes=num_region_classes, num_severity_classes=num_severity_classes, dropout=dropout)
        
        # Free-form answer decoder
        self.answer_decoder = AnswerDecoder(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=decoder_layers,
            num_heads=num_attention_heads,
            max_length=max_answer_length,
            dropout=dropout
        )
        
        # Template-based answer generator
        self.template_generator = TemplateAnswerGenerator()
        
        # CheXpert head
        self.chexpert_head = CheXpertHead(hidden_size=hidden_size, num_classes=14, dropout=dropout) if use_chexpert_head else None
        
        # Scene Graph Generator (can be frozen during finetune)
        self.scene_graph_generator = SceneGraphGenerator(
            visual_dim=self.visual_encoder.backbone_dim,
            hidden_size=hidden_size,
            num_entity_classes=num_entities,
            num_region_classes=num_regions,
            num_relationships=10,
            max_objects=20,
            dropout=dropout
        )
        
        # Visual Grounding Head (scene graph-guided)
        self.grounding_head = VisualGroundingHead(hidden_size=hidden_size, num_visual_features=49, dropout=dropout)
        
        # Tokenizer for decoding (loaded lazily)
        self._tokenizer = None
        
        self.set_training_mode(training_mode)
    
    def set_training_mode(self, mode: str):
        """Configure trainable parameters based on mode."""
        self.training_mode = mode
        if mode == 'standard':
            for p in self.scene_graph_generator.parameters():
                p.requires_grad = False
            for p in self.grounding_head.parameters():
                p.requires_grad = False
        elif mode == 'finetune':
            for p in self.scene_graph_generator.parameters():
                p.requires_grad = False
            for p in self.grounding_head.parameters():
                p.requires_grad = True
        elif mode in ('pretrain', 'full'):
            for p in self.parameters():
                p.requires_grad = True
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        if hasattr(self.text_encoder.encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.encoder.gradient_checkpointing_enable()
        # Enable for mHC block
        if self.mhc_fusion is not None:
            self.mhc_fusion._gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        if hasattr(self.text_encoder, 'encoder'):
            if hasattr(self.text_encoder.encoder, 'gradient_checkpointing_disable'):
                self.text_encoder.encoder.gradient_checkpointing_disable()
            if hasattr(self.text_encoder.encoder, 'gradient_checkpointing'):
                self.text_encoder.encoder.gradient_checkpointing = False
        # Disable for mHC block
        if self.mhc_fusion is not None:
            self.mhc_fusion._gradient_checkpointing = False
    
    def _get_model_dtype(self) -> torch.dtype:
        """
        Get the authoritative dtype for this model's parameters.
        
        With DeepSpeed FP16, parameters are converted to fp16. However, some
        operations (BatchNorm, LayerNorm, etc.) may upcast intermediate tensors
        to fp32. This method returns the dtype that Linear layers expect.
        """
        try:
            return next(self.visual_proj.parameters()).dtype
        except StopIteration:
            return torch.float32
    
    def _ensure_dtype(self, tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        """Cast tensor to target dtype if needed (no-op if already correct)."""
        if tensor.dtype != target_dtype:
            return tensor.to(dtype=target_dtype)
        return tensor
    
    def _sg_outputs_to_dicts(
        self,
        sg_outputs: Dict[str, Any],
        objectness_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Convert SceneGraphGenerator outputs to the dict format that
        SceneGraphEncoder expects.  This closes the inference gap:
        the model generates a scene graph from the image and then
        feeds it back into the multimodal fusion pipeline.
        
        Args:
            sg_outputs: Dict from SceneGraphGenerator.forward()
            objectness_threshold: Min score to keep a detected object
            
        Returns:
            List[dict] with keys: bboxes, region_ids, entity_ids,
            positiveness, num_objects  (one per batch element)
        """
        bbox_preds = sg_outputs['bbox_preds']           # (B, N, 4)
        entity_logits = sg_outputs['entity_logits']     # (B, N, num_entities)
        region_logits = sg_outputs['region_logits']      # (B, N, num_regions)
        objectness = sg_outputs['objectness_scores']     # (B, N)
        positiveness_logits = sg_outputs['positiveness_logits']  # (B, N, 2)
        
        B, N = bbox_preds.shape[:2]
        scene_graph_dicts = []
        
        for b in range(B):
            # Determine which objects pass the threshold
            scores = objectness[b]  # (N,)
            keep = scores >= objectness_threshold
            n_keep = int(keep.sum().item())
            
            if n_keep == 0:
                # At least 1 object (whole-image fallback)
                n_keep = 1
                keep = torch.zeros(N, dtype=torch.bool, device=bbox_preds.device)
                keep[0] = True
            
            sg_dict = {
                'bboxes': bbox_preds[b, keep].detach().cpu().numpy(),
                'entity_ids': entity_logits[b, keep].argmax(dim=-1).detach().cpu().numpy(),
                'region_ids': region_logits[b, keep].argmax(dim=-1).detach().cpu().numpy(),
                'positiveness': positiveness_logits[b, keep].argmax(dim=-1).detach().cpu().numpy(),
                'num_objects': n_keep,
            }
            scene_graph_dicts.append(sg_dict)
        
        return scene_graph_dicts
    
    def get_mhc_metrics(self) -> Dict[str, float]:
        """Get mHC-specific metrics for logging (path weights, gate values)."""
        if self.mhc_fusion is not None:
            return self.mhc_fusion.get_metrics()
        return {}
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer for answer decoding."""
        if self._tokenizer is None and TRANSFORMERS_AVAILABLE:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
            except:
                pass
        return self._tokenizer
    
    def decode_generated_ids(self, generated_ids: torch.Tensor) -> List[str]:
        """Decode generated token IDs to text."""
        if self.tokenizer is None:
            return ["[Tokenizer not available]"] * generated_ids.shape[0]
        
        texts = []
        for ids in generated_ids:
            # Remove special tokens and decode
            ids_list = ids.tolist()
            # Stop at EOS
            if self.answer_decoder.eos_token_id in ids_list:
                ids_list = ids_list[:ids_list.index(self.answer_decoder.eos_token_id)]
            # Remove BOS
            if ids_list and ids_list[0] == self.answer_decoder.bos_token_id:
                ids_list = ids_list[1:]
            
            text = self.tokenizer.decode(ids_list, skip_special_tokens=True)
            texts.append(text)
        
        return texts
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scene_graphs: List[Dict[str, Any]],
        token_type_ids: Optional[torch.Tensor] = None,
        question_types: Optional[List[str]] = None,
        image_widths: Optional[torch.Tensor] = None,
        image_heights: Optional[torch.Tensor] = None,
        view_encodings: Optional[torch.Tensor] = None,  # (B, 4) one-hot [PA, AP, LATERAL, OTHER]
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_entities: Optional[List[torch.Tensor]] = None,
        gt_regions: Optional[List[torch.Tensor]] = None,
        answer_ids: Optional[torch.Tensor] = None,  # Target answer tokens for training decoder
    ) -> Dict[str, Any]:
        """
        Forward pass returning all outputs.
        
        Args:
            images: (B, 3, H, W) input images
            input_ids: (B, L) question token IDs
            attention_mask: (B, L) question attention mask
            scene_graphs: List of scene graph dicts
            token_type_ids: Optional (B, L) token type IDs
            question_types: Optional list of question type strings
            view_encodings: Optional (B, 4) one-hot view position encoding from MIMIC-CXR-JPG
                           [PA, AP, LATERAL, OTHER] - useful for view-aware processing
            answer_ids: Optional (B, T) target answer tokens for decoder training
            
        Returns:
            Dict with all outputs including generated answers
        """
        device = images.device
        batch_size = images.shape[0]
        local_scene_graphs = scene_graphs[:batch_size]
        
        # =====================================================================
        # DTYPE MANAGEMENT: DeepSpeed FP16 converts params to fp16, but some
        # PyTorch ops (BatchNorm, LayerNorm, etc.) may upcast to fp32.
        # We must ensure all tensors match param dtype before each Linear layer.
        # =====================================================================
        dt = self._get_model_dtype()  # Authoritative dtype from model params
        
        # Cast images to model dtype
        images = self._ensure_dtype(images, dt)
        
        # Get bboxes from scene graphs
        bboxes = [torch.tensor(sg['bboxes'], dtype=dt, device=device) for sg in local_scene_graphs]
        
        # --- VISUAL BACKBONE ---
        feature_maps = self._ensure_dtype(self.visual_encoder.get_feature_maps(images), dt)
        encoder_output = self._ensure_dtype(self.visual_encoder(images, bboxes), dt)
        visual_features = self.visual_proj(encoder_output)
        visual_features = self._ensure_dtype(visual_features, dt)  # Guard after projection
        
        # Visual mask
        visual_mask = torch.zeros(batch_size, visual_features.shape[1], dtype=dt, device=device)
        for i, sg in enumerate(local_scene_graphs):
            num_objects = min(sg['num_objects'], visual_features.shape[1])
            visual_mask[i, :num_objects] = 1.0
        
        # --- SCENE GRAPH GENERATION ---
        scene_graph_outputs = self.scene_graph_generator(feature_maps, gt_bboxes, gt_entities, gt_regions)
        
        # --- TEXT ENCODING ---
        text_features, text_pooled = self.text_encoder(input_ids, attention_mask, token_type_ids)
        text_features = self._ensure_dtype(text_features, dt)
        text_pooled = self._ensure_dtype(text_pooled, dt)
        
        # --- SCENE GRAPH ENCODING ---
        # At inference (not training), use the GENERATED scene graph from the
        # SceneGraphGenerator instead of the input scene graphs (which may be
        # dummies).  This closes the inference gap: image → SG Generator →
        # SG Encoder → multimodal fusion.
        # During training we keep using the dataset's ground-truth scene graphs
        # so the encoder learns from high-quality supervision.
        if not self.training:
            sg_dicts_for_encoder = self._sg_outputs_to_dicts(scene_graph_outputs)
        else:
            sg_dicts_for_encoder = local_scene_graphs
        
        scene_features, scene_mask = self.scene_encoder(sg_dicts_for_encoder, device)
        scene_features = self._ensure_dtype(scene_features, dt)
        scene_features = self.scene_proj(scene_features)
        scene_features = self._ensure_dtype(scene_features, dt)  # Guard after projection
        
        # --- MULTIMODAL FUSION (SIM) ---
        text_mask = attention_mask.to(dtype=dt)
        scene_mask_dt = scene_mask.to(dtype=dt)
        fused, attention_weights, text_hidden = self.sim(
            visual_features, text_features, scene_features,
            visual_mask=visual_mask, text_mask=text_mask, scene_mask=scene_mask_dt
        )
        # LayerNorm/attention may upcast - ensure fused stays in dt
        fused = self._ensure_dtype(fused, dt)
        text_hidden = self._ensure_dtype(text_hidden, dt)
        
        # --- mHC FUSION (if enabled) ---
        if self.mhc_fusion is not None:
            fused_seq = fused.unsqueeze(1)  # (B, 1, D)
            fused_seq = self.mhc_fusion(fused_seq)
            fused = self._ensure_dtype(fused_seq.squeeze(1), dt)  # (B, D)
        
        # --- VISUAL GROUNDING (scene graph-guided) ---
        grounding_outputs = self.grounding_head(
            self._ensure_dtype(fused, dt),
            self._ensure_dtype(visual_features, dt),
            visual_mask,
            scene_features=self._ensure_dtype(scene_features, dt),
            scene_mask=scene_mask,
        )
        
        # --- CLASSIFICATION VQA ---
        vqa_logits = self.answer_module(self._ensure_dtype(fused, dt))
        
        # --- FREE-FORM ANSWER GENERATION (PRIMARY - trained on MIMIC-Ext-CXR-QBA) ---
        decoder_outputs = self.answer_decoder(
            self._ensure_dtype(fused, dt),
            self._ensure_dtype(text_hidden, dt),
            target_ids=answer_ids,
            encoder_mask=attention_mask
        )
        generated_ids = decoder_outputs['generated_ids']
        generation_logits = decoder_outputs['logits']
        
        # Decode to text (inference only)
        generated_text = self.decode_generated_ids(generated_ids) if answer_ids is None else None
        
        # --- TEMPLATE ANSWERS (FALLBACK) ---
        template_answers = self.template_generator(vqa_logits, question_types)
        
        # --- CHEXPERT PREDICTION ---
        chexpert_logits = None
        if self.chexpert_head:
            vm = visual_mask.unsqueeze(-1).to(dtype=dt)
            visual_pooled = (visual_features * vm).sum(1) / vm.sum(1).clamp(min=1)
            chexpert_logits = self.chexpert_head(self._ensure_dtype(visual_pooled, dt))
        
        # Get mHC metrics if enabled
        mhc_metrics = self.get_mhc_metrics() if self.use_mhc else {}
        
        return {
            # === CLASSIFICATION OUTPUTS ===
            'vqa_logits': vqa_logits,           # Dict: binary/category/region/severity logits
            'chexpert_logits': chexpert_logits, # (B, 14) CheXpert auxiliary predictions
            'pooled_output': fused,             # (B, D) fused multimodal representation
            
            # === ANSWER GENERATION (PRIMARY - from decoder trained on MIMIC-Ext-CXR-QBA) ===
            'generated_answer_ids': generated_ids,     # (B, T) generated token IDs
            'generated_answer_logits': generation_logits,  # (B, T, V) vocab logits for loss
            'generated_answer_text': generated_text,   # List[str] - REPORT-QUALITY ANSWERS
            # ^ This is the PRIMARY answer output - trained on actual hierarchical 
            # answers from MIMIC-Ext-CXR-QBA dataset (42M+ report-derived QA pairs)
            
            # === TEMPLATE ANSWERS (FALLBACK - from classification heads) ===
            'template_answer': template_answers,  # List[str] - simple deterministic fallback
            # ^ Only use as baseline/fallback - NOT report quality
            
            # === EXPLAINABILITY ===
            'attention_weights': attention_weights,  # Dict: text_to_visual, visual_to_text, etc.
            
            # === SCENE GRAPH GENERATION ===
            'scene_graph_outputs': scene_graph_outputs,  # Dict: bbox_preds, entity_logits, etc.
            # At inference the generated SG dicts (from SceneGraphGenerator) are
            # fed back into SceneGraphEncoder for fusion.  Expose them here so
            # callers can inspect the predicted scene graph.
            'generated_scene_graphs': sg_dicts_for_encoder if not self.training else None,
            
            # === VISUAL GROUNDING ===
            'grounding_outputs': grounding_outputs,  # Dict: bbox_pred, pointing_score, etc.
            
            # === mHC METRICS (for stability analysis) ===
            'mhc_metrics': mhc_metrics,  # Dict: path weights, gate values, amax_gain
        }
    
    def generate_answer(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scene_graphs: List[Dict[str, Any]],
        question_types: Optional[List[str]] = None,
        use_decoder: bool = True,
        use_template: bool = False,  # Default to False - decoder is primary
    ) -> Dict[str, List[str]]:
        """
        Convenience method for answer generation at inference time.
        
        The decoder generates report-quality answers (trained on MIMIC-Ext-CXR-QBA).
        Template answers are a simple fallback from classification outputs.
        
        Args:
            images: (B, 3, H, W) input images
            input_ids: (B, L) tokenized question
            attention_mask: (B, L) question attention mask
            scene_graphs: List of scene graph dicts
            question_types: Optional question type strings for template selection
            use_decoder: Include decoder output (PRIMARY - report-quality answers)
            use_template: Include template output (FALLBACK - simple answers)
            
        Returns:
            Dict with:
            - 'decoder_answer': List[str] - REPORT-QUALITY answers from decoder
            - 'template_answer': List[str] - Simple fallback from classification
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, input_ids, attention_mask, scene_graphs, question_types=question_types)
        
        result = {}
        if use_decoder:
            result['decoder_answer'] = outputs['generated_answer_text']
        if use_template:
            result['template_answer'] = outputs['template_answer']
        
        return result
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load model from checkpoint."""
        import json
        from pathlib import Path
        
        path = Path(path)
        config_path = path / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            model_config = config.get('model', config)
            for key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'training_mode', 'use_chexpert_head', 'vocab_size', 'max_answer_length']:
                if key in model_config and key not in kwargs:
                    kwargs[key] = model_config[key]
        
        model = cls(**kwargs)
        weights_path = path / 'pytorch_model.bin'
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        return model
    
    def save_pretrained(self, path: str):
        """Save model to checkpoint."""
        import json
        from pathlib import Path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / 'pytorch_model.bin')
        
        config = {
            'hidden_size': self.hidden_size,
            'use_chexpert_head': self.use_chexpert_head,
            'training_mode': self.training_mode,
            'vocab_size': self.vocab_size,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_visual_attention_map(attention_dict: Dict[str, torch.Tensor], spatial_size: Tuple[int, int] = (7, 7)) -> Optional[torch.Tensor]:
    """Extract 2D spatial attention map from attention weights."""
    attn = attention_dict.get('text_to_visual')
    if attn is None:
        return None
    
    B, H, T, V = attn.shape
    attn = attn.mean(dim=1)[:, 0]
    
    sqrt_v = int(V ** 0.5)
    if sqrt_v * sqrt_v == V:
        return attn.view(B, sqrt_v, sqrt_v)
    return F.interpolate(attn.view(B, 1, -1), size=spatial_size[0] * spatial_size[1], mode='linear').view(B, *spatial_size)


def attention_to_bbox(attention_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert attention map to bounding box."""
    B, H, W = attention_map.shape
    device = attention_map.device
    
    attn_min = attention_map.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
    attn_max = attention_map.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
    attention_map = (attention_map - attn_min) / (attn_max - attn_min + 1e-8)
    
    bboxes = torch.zeros(B, 4, device=device)
    for b in range(B):
        mask = (attention_map[b] >= threshold).float()
        if mask.sum() == 0:
            flat_idx = attention_map[b].argmax()
            y, x = flat_idx // W, flat_idx % W
            bboxes[b] = torch.tensor([x/W - 0.1, y/H - 0.1, x/W + 0.1, y/H + 0.1], device=device)
        else:
            rows, cols = mask.any(dim=1), mask.any(dim=0)
            y_indices, x_indices = torch.where(rows)[0], torch.where(cols)[0]
            y1, y2 = y_indices[0].item(), y_indices[-1].item()
            x1, x2 = x_indices[0].item(), x_indices[-1].item()
            bboxes[b] = torch.tensor([x1/W, y1/H, (x2+1)/W, (y2+1)/H], device=device)
    
    return bboxes.clamp(0, 1)
