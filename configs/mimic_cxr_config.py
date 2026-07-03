"""
MIMIC-CXR VQA Configuration

Centralized configuration for all training, model, and data parameters.
Based on MIMIC_CXR_VQA_ANALYSIS.md specifications.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Backbones
    visual_backbone: str = "convnext_base"
    text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT"
    
    # Feature dimensions
    visual_feature_dim: int = 512
    scene_graph_dim: int = 134  # 6 bbox + 64 region_emb + 64 entity_emb
    visual_embedding_dim: int = 646  # scene_graph_dim + visual_feature_dim
    
    # VisualBERT architecture
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Scene-Embedded Interaction Module
    sim_layers: int = 2
    
    # Scene graph vocabulary
    num_regions: int = 310  # MIMIC-Ext-CXR-QBA anatomical regions
    num_entities: int = 237  # MIMIC-Ext-CXR-QBA finding entities
    region_embedding_dim: int = 64
    entity_embedding_dim: int = 64
    
    # Answer heads
    num_binary_classes: int = 2  # Yes/No
    num_category_classes: int = 14  # CheXpert categories
    num_region_classes: int = 26  # Major anatomical regions
    num_severity_classes: int = 4  # None, Mild, Moderate, Severe
    
    # Text processing
    max_question_length: int = 128
    vocab_size: int = 30522  # BERT vocab size

    # ----- V2 (Qwen VLM) keys -----
    qwen_model_id: Optional[str] = None
    use_quantization: Optional[bool] = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: Optional[List[str]] = None
    num_sg_tokens: int = 8

    # ----- V2 standalone scene-graph generator -----
    # When non-None, train_mimic_cxr.py dispatches to the standalone SG
    # generator trainer at phase=sg_only, and swaps the loaded generator
    # into SSGVQANetV2.sg_generator at phases 2-4. Shape:
    #   sg_generator:
    #     name: txrv_detr
    #     num_entities: 22
    #     num_regions: 30
    #     checkpoint: ./checkpoints/.../best_model   # frozen load in stages 2-4
    #     frozen: true
    sg_generator: Optional[Dict[str, Any]] = None


@dataclass
class DataConfig:
    """Data loading configuration."""
    # Dataset paths
    mimic_cxr_jpg_path: str = "/path/to/MIMIC-CXR-JPG"
    mimic_ext_cxr_qba_path: str = "/path/to/MIMIC-Ext-CXR-QBA"
    chexpert_labels_path: str = ""  # Optional: path to chexpert.csv
    test_labels_csv_path: str = ""  # Optional: path to test labels
    
    # Quality filtering
    quality_grade: str = "A"  # A for fine-tuning, B for pre-training
    view_filter: str = "frontal_only"  # frontal_only, lateral_only, all
    
    # Question type filtering (None = all types)
    question_types: Optional[List[str]] = None
    
    # Pre-filtered exports (faster loading for large datasets)
    use_exports: bool = False  # Use exports/ folder with pre-filtered data
    export_grade: str = ""  # e.g., "B_frontal" for exports/B_frontal/
    
    # Caching for distributed training (prevents NCCL timeout)
    cache_dir: str = ".cache/dataset_samples"  # Cache directory for samples
    
    # Image preprocessing
    image_size: int = 224
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # ----- V2 additions -----
    # Question-type blacklist applied AFTER the whitelist (question_types).
    # Stage 4 drops A_indication/A_history/A_technique/A_comparison because
    # those ask about report free text, not the image.
    skip_question_types: Optional[List[str]] = None

    # Drop scene-graph observations whose QBA localization_quality is below
    # this level. Stage 1 sets it to 3 (BBOX_LOCALIZATION); leaves 0 elsewhere
    # so VQA training still sees all observations.
    min_localization_quality: int = 0

    # Cap on val dataset size when not using --max_samples. Bumped from
    # the implicit 250 to 1000 so a single eval can distinguish runs with
    # better than +/-6pp 95% CI.
    val_samples: Optional[int] = None

    # Override quality_grade for the val dataset only. Used by Stage 4's
    # train-B/val-A pattern when we want a wider train pool but still
    # measure on the target A distribution.
    val_quality_grade: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Output
    output_dir: str = "./checkpoints/mimic-cxr-vqa"
    
    # Training phase (pretrain or finetune)
    phase: str = "pretrain"  # pretrain or finetune
    
    # Batch size
    batch_size_per_gpu: int = 16
    gradient_accumulation_steps: int = 4
    
    # Learning rate
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Epochs
    num_epochs: int = 20
    
    # Mixed precision
    fp16: bool = True
    
    # Gradient checkpointing (disabled by default - causes deadlock with DataParallel)
    gradient_checkpointing: bool = False
    
    # Loss weights (top-level task weights)
    vqa_loss_weight: float = 1.0
    generation_loss_weight: float = 0.5
    chexpert_loss_weight: float = 0.3
    scene_graph_loss_weight: float = 0.1
    grounding_loss_weight: float = 0.1
    # Per-head VQA classification weights
    binary_head_weight: float = 1.0
    category_head_weight: float = 0.5
    region_head_weight: float = 0.5
    severity_head_weight: float = 0.3
    
    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 5000
    save_total_limit: int = 5
    eval_steps: int = 2500
    
    # Best model tracking
    metric_for_best_model: str = "classification_accuracy"
    greater_is_better: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    
    # Data loading - optimized for high-CPU machines
    dataloader_num_workers: int = 8  # Per GPU
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True  # Keep workers alive between epochs
    
    # Hugging Face Hub
    hub_model_id: str = ""
    hub_private_repo: bool = True
    push_to_hub_strategy: str = "best"  # best, checkpoint, end
    push_every_save: bool = False  # If True, push to HF on every save (not just best)
    
    # Reproducibility
    seed: int = 42

    # Exponential moving average of weights, used for validation and best-model
    # save. 0.0 = disabled (backwards-compat default). 0.999 is a reasonable
    # value for LoRA fine-tuning at Stage 3/4.
    ema_decay: float = 0.0


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = True
    project: str = "mimic-cxr-vqa"
    entity: str = ""  # Optional: your wandb username or team
    name: str = ""  # Optional: run name (auto-generated if empty)
    group: str = "experiments"
    tags: List[str] = field(default_factory=lambda: ["ssg-vqa", "mimic-cxr", "medical-vqa"])
    notes: str = "SSG-VQA-Net adapted for MIMIC-CXR chest X-ray VQA"
    
    # Model watching
    watch_model: bool = False
    watch_log_freq: int = 1000
    
    # Artifact logging
    log_model: bool = True


@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration."""
    enabled: bool = False
    config_path: str = "configs/deepspeed_config.json"
    stage: int = 2  # ZeRO Stage


@dataclass
class MIMICCXRVQAConfig:
    """
    Complete configuration for MIMIC-CXR VQA training.
    
    Usage:
        config = MIMICCXRVQAConfig()
        config.training.learning_rate = 1e-4
        
        # Or load from file
        config = load_config_from_file("config.yaml")
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'wandb': asdict(self.wandb),
            'deepspeed': asdict(self.deepspeed),
        }
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MIMICCXRVQAConfig":
        """Create config from dictionary. Silently drops keys not in the
        declared dataclass fields so YAMLs can carry annotations, legacy v1
        keys, and future experimental toggles without blowing up the loader.
        A warning is logged for each dropped key so mistakes still surface.
        """
        import dataclasses
        import logging as _logging
        _log = _logging.getLogger(__name__)

        def _filter(dc_cls, raw: Dict[str, Any]) -> Dict[str, Any]:
            known = {f.name for f in dataclasses.fields(dc_cls)}
            kept: Dict[str, Any] = {}
            for k, v in (raw or {}).items():
                if k in known:
                    kept[k] = v
                else:
                    _log.info(
                        f"[config] dropping unknown {dc_cls.__name__} key "
                        f"'{k}' (not a declared field). This is safe if the "
                        f"trainer reads the key via getattr(config.*, ...)."
                    )
            return kept

        return cls(
            model=ModelConfig(**_filter(ModelConfig, config_dict.get('model', {}))),
            data=DataConfig(**_filter(DataConfig, config_dict.get('data', {}))),
            training=TrainingConfig(**_filter(TrainingConfig, config_dict.get('training', {}))),
            wandb=WandbConfig(**_filter(WandbConfig, config_dict.get('wandb', {}))),
            deepspeed=DeepSpeedConfig(**_filter(DeepSpeedConfig, config_dict.get('deepspeed', {}))),
        )


def get_default_config() -> MIMICCXRVQAConfig:
    """Get default configuration."""
    return MIMICCXRVQAConfig()


def load_config_from_file(path: str) -> MIMICCXRVQAConfig:
    """Load configuration from YAML or JSON file."""
    with open(path, 'r') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)
    
    return MIMICCXRVQAConfig.from_dict(config_dict)


# Preset configurations for different scenarios
def get_debug_config() -> MIMICCXRVQAConfig:
    """Get configuration for debugging (small batch, few epochs)."""
    config = MIMICCXRVQAConfig()
    config.training.batch_size_per_gpu = 2
    config.training.num_epochs = 2
    config.training.logging_steps = 10
    config.training.save_steps = 100
    config.training.eval_steps = 50
    config.wandb.enabled = False
    return config


def get_pretrain_config() -> MIMICCXRVQAConfig:
    """Get configuration for pre-training phase."""
    config = MIMICCXRVQAConfig()
    config.data.quality_grade = "B"  # B and above
    config.training.learning_rate = 1e-4  # Slightly higher for pretraining
    config.training.num_epochs = 10 # Fewer epochs for pretraining
    config.training.chexpert_loss_weight = 0.5  # Heavier auxiliary supervision
    return config


def get_finetune_config() -> MIMICCXRVQAConfig:
    """Get configuration for fine-tuning phase."""
    config = MIMICCXRVQAConfig()
    config.data.quality_grade = "A"  # A grade only
    config.training.learning_rate = 2e-5  # Conservative for finetuning
    config.training.num_epochs = 20
    config.training.chexpert_loss_weight = 0.3  # Lighter auxiliary
    return config
