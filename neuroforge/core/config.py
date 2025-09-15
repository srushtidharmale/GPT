"""Configuration management for NeuroForge."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    type: str  # "retnet", "moe", "transformer", "multimodal"
    vocab_size: int = 50257
    embed_dim: int = 1536
    num_heads: int = 12
    num_layers: int = 24
    max_seq_len: int = 2048
    dropout: float = 0.1
    use_flash_attn: bool = True
    use_gradient_checkpointing: bool = True
    
    # RetNet specific
    gamma: float = 1.0
    use_retention: bool = True
    
    # MoE specific
    num_experts: int = 8
    expert_capacity: int = 4
    router_jitter_noise: float = 0.1
    
    # Multi-modal specific
    vision_dim: int = 512
    audio_dim: int = 256
    fusion_method: str = "cross_attention"  # "cross_attention", "concat", "gated"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    learning_rate: float = 6e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Advanced training
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    use_qlora: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    
    # Mixed precision
    use_fp16: bool = True
    use_bf16: bool = False
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    
    # Streaming
    stream_chunk_size: int = 10
    stream_delay: float = 0.05
    
    # Memory optimization
    use_kv_cache: bool = True
    kv_cache_size: int = 1024


@dataclass
class DataConfig:
    """Configuration for data processing."""
    dataset_name: str = "fineweb-edu"
    dataset_config: str = "default"
    max_samples: Optional[int] = None
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    
    # Tokenization
    tokenizer_name: str = "gpt2"
    custom_tokenizer_path: Optional[str] = None
    
    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class SystemConfig:
    """System-level configuration."""
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    num_gpus: int = 1
    distributed: bool = False
    mixed_precision: bool = True
    
    # Memory management
    max_memory_usage: float = 0.9
    gradient_checkpointing: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "neuroforge"


@dataclass
class NeuroForgeConfig:
    """Main configuration class for NeuroForge."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Paths
    model_dir: str = "./models"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    output_dir: str = "./outputs"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Convert string paths to Path objects
        self.model_dir = Path(self.model_dir)
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        
        # Create directories if they don't exist
        for path in [self.model_dir, self.data_dir, self.cache_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "NeuroForgeConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse nested configurations
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        inference_config = InferenceConfig(**config_dict.get('inference', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            inference=inference_config,
            data=data_config,
            system=system_config,
            **config_dict.get('paths', {})
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'data': self.data.__dict__,
            'system': self.system.__dict__,
            'paths': {
                'model_dir': str(self.model_dir),
                'data_dir': str(self.data_dir),
                'cache_dir': str(self.cache_dir),
                'output_dir': str(self.output_dir),
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
