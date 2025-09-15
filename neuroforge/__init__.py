"""
NeuroForge - Advanced Multi-Modal AI Platform

A next-generation AI platform combining multiple modalities with state-of-the-art
transformer architectures and real-time capabilities.
"""

__version__ = "1.0.0"
__author__ = "NeuroForge Team"
__email__ = "team@neuroforge.ai"

from .core.engine import NeuroForgeEngine
from .models.retnet import RetNetModel
from .models.moe import MixtureOfExpertsModel
from .models.multimodal import MultiModalModel
from .training.trainer import AdvancedTrainer
from .inference.streaming import StreamingInference

__all__ = [
    "NeuroForgeEngine",
    "RetNetModel", 
    "MixtureOfExpertsModel",
    "MultiModalModel",
    "AdvancedTrainer",
    "StreamingInference",
]
