"""
NeuroForge Core Engine

The main engine that orchestrates all components of the NeuroForge platform,
including model management, inference, and multi-modal processing.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import logging
from pathlib import Path
import json
import time
from dataclasses import asdict

from .config import NeuroForgeConfig
from ..models.retnet import RetNetModel
from ..models.moe import MixtureOfExpertsModel
from ..models.multimodal import MultiModalModel
from ..inference.streaming import StreamingInference
from ..training.trainer import AdvancedTrainer


logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing different model types."""
    
    MODEL_TYPES = {
        "retnet": RetNetModel,
        "moe": MixtureOfExpertsModel,
        "multimodal": MultiModalModel,
    }
    
    @classmethod
    def get_model_class(cls, model_type: str) -> nn.Module:
        """Get model class by type."""
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")
        return cls.MODEL_TYPES[model_type]
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model types."""
        return list(cls.MODEL_TYPES.keys())


class NeuroForgeEngine:
    """
    Main NeuroForge engine that manages models, inference, and multi-modal processing.
    
    This is the central component that coordinates all aspects of the NeuroForge platform.
    """
    
    def __init__(self, config: NeuroForgeConfig):
        self.config = config
        self.device = self._setup_device()
        self.models: Dict[str, nn.Module] = {}
        self.streaming_inference: Optional[StreamingInference] = None
        self.trainer: Optional[AdvancedTrainer] = None
        
        # Initialize logging
        self._setup_logging()
        
        # Load models if they exist
        self._load_existing_models()
        
        logger.info(f"NeuroForge Engine initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.system.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.system.device)
        
        return device
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.system.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.system.log_file) if self.config.system.log_file else logging.NullHandler()
            ]
        )
    
    def _load_existing_models(self):
        """Load existing models from disk."""
        model_dir = self.config.model_dir
        if not model_dir.exists():
            return
        
        for model_file in model_dir.glob("*.pt"):
            try:
                model_name = model_file.stem
                self.load_model(model_name, str(model_file))
                logger.info(f"Loaded existing model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_file}: {e}")
    
    def create_model(
        self,
        model_name: str,
        model_type: str,
        **kwargs
    ) -> nn.Module:
        """
        Create a new model instance.
        
        Args:
            model_name: Name for the model
            model_type: Type of model to create ("retnet", "moe", "multimodal")
            **kwargs: Additional model parameters
            
        Returns:
            Created model instance
        """
        if model_name in self.models:
            raise ValueError(f"Model {model_name} already exists")
        
        # Get model class
        model_class = ModelRegistry.get_model_class(model_type)
        
        # Merge config with kwargs
        model_config = asdict(self.config.model)
        model_config.update(kwargs)
        
        # Create model instance
        model = model_class(**model_config)
        model = model.to(self.device)
        
        # Store model
        self.models[model_name] = model
        
        logger.info(f"Created model {model_name} of type {model_type}")
        logger.info(f"Model parameters: {model.get_num_params():,}")
        
        return model
    
    def load_model(self, model_name: str, model_path: str) -> nn.Module:
        """
        Load a model from disk.
        
        Args:
            model_name: Name for the model
            model_path: Path to model checkpoint
            
        Returns:
            Loaded model instance
        """
        if model_name in self.models:
            raise ValueError(f"Model {model_name} already exists")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model type from checkpoint
        model_type = checkpoint.get('model_type', 'retnet')
        model_class = ModelRegistry.get_model_class(model_type)
        
        # Create model instance
        model_config = checkpoint.get('config', {})
        model = model_class(**model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Store model
        self.models[model_name] = model
        
        logger.info(f"Loaded model {model_name} from {model_path}")
        
        return model
    
    def save_model(self, model_name: str, save_path: Optional[str] = None) -> str:
        """
        Save a model to disk.
        
        Args:
            model_name: Name of the model to save
            save_path: Optional custom save path
            
        Returns:
            Path where model was saved
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if save_path is None:
            save_path = self.config.model_dir / f"{model_name}.pt"
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_type': self._get_model_type(model),
            'config': asdict(self.config.model),
            'timestamp': time.time(),
            'device': str(self.device)
        }
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        
        logger.info(f"Saved model {model_name} to {save_path}")
        
        return str(save_path)
    
    def _get_model_type(self, model: nn.Module) -> str:
        """Get model type from model instance."""
        if isinstance(model, RetNetModel):
            return "retnet"
        elif isinstance(model, MixtureOfExpertsModel):
            return "moe"
        elif isinstance(model, MultiModalModel):
            return "multimodal"
        else:
            return "unknown"
    
    def get_model(self, model_name: str) -> nn.Module:
        """Get a model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name]
    
    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())
    
    def delete_model(self, model_name: str):
        """Delete a model from memory."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        del self.models[model_name]
        logger.info(f"Deleted model {model_name}")
    
    def setup_streaming_inference(self, model_name: str) -> StreamingInference:
        """
        Setup streaming inference for a model.
        
        Args:
            model_name: Name of the model to use for inference
            
        Returns:
            StreamingInference instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        self.streaming_inference = StreamingInference(
            model=model,
            config=self.config.inference,
            device=self.device
        )
        
        logger.info(f"Setup streaming inference for model {model_name}")
        
        return self.streaming_inference
    
    def setup_trainer(self, model_name: str) -> AdvancedTrainer:
        """
        Setup trainer for a model.
        
        Args:
            model_name: Name of the model to train
            
        Returns:
            AdvancedTrainer instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        self.trainer = AdvancedTrainer(
            model=model,
            config=self.config.training,
            device=self.device
        )
        
        logger.info(f"Setup trainer for model {model_name}")
        
        return self.trainer
    
    async def generate_text(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using a model.
        
        Args:
            model_name: Name of the model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            stream: Whether to stream the output
            
        Returns:
            Generated text or async generator for streaming
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if stream:
            return self._generate_text_stream(
                model, prompt, max_tokens, temperature, top_k, top_p
            )
        else:
            return await self._generate_text_batch(
                model, prompt, max_tokens, temperature, top_k, top_p
            )
    
    async def _generate_text_batch(
        self,
        model: nn.Module,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: float
    ) -> str:
        """Generate text in batch mode."""
        # Tokenize prompt
        # Note: In a real implementation, you would use a proper tokenizer
        # For now, we'll simulate tokenization
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)  # Placeholder
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode generated text
        # Note: In a real implementation, you would use a proper tokenizer
        generated_text = f"Generated text for prompt: {prompt}"  # Placeholder
        
        return generated_text
    
    async def _generate_text_stream(
        self,
        model: nn.Module,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: float
    ) -> AsyncGenerator[str, None]:
        """Generate text in streaming mode."""
        # Tokenize prompt
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)  # Placeholder
        
        # Stream generation
        for i in range(max_tokens):
            # Simulate token generation
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Generate next token
            with torch.no_grad():
                # Placeholder for actual generation
                next_token = f"token_{i}"
            
            yield next_token
    
    async def process_multimodal(
        self,
        model_name: str,
        text: Optional[str] = None,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multi-modal inputs.
        
        Args:
            model_name: Name of the model to use
            text: Text input
            image: Image input
            audio: Audio input
            **kwargs: Additional parameters
            
        Returns:
            Processing results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if not isinstance(model, MultiModalModel):
            raise ValueError(f"Model {model_name} is not a multi-modal model")
        
        # Prepare inputs
        input_ids = None
        if text:
            # Tokenize text
            input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)  # Placeholder
        
        # Process with model
        with torch.no_grad():
            logits, loss, info = model(
                input_ids=input_ids,
                images=image,
                audio_features=audio
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'info': info
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        return {
            'name': model_name,
            'type': self._get_model_type(model),
            'parameters': model.get_num_params(),
            'device': str(self.device),
            'config': asdict(self.config.model)
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'models_loaded': len(self.models),
            'config': asdict(self.config.system)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the engine."""
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'models': {},
            'system': self.get_system_info()
        }
        
        # Check each model
        for model_name in self.models:
            try:
                model = self.models[model_name]
                # Simple forward pass to check model health
                dummy_input = torch.tensor([[1, 2, 3]], device=self.device)
                with torch.no_grad():
                    if isinstance(model, MultiModalModel):
                        model(dummy_input)
                    else:
                        model(dummy_input)
                
                health_status['models'][model_name] = 'healthy'
            except Exception as e:
                health_status['models'][model_name] = f'unhealthy: {str(e)}'
                health_status['status'] = 'degraded'
        
        return health_status


# Async generator type hint
from typing import AsyncGenerator
