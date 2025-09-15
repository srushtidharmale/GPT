"""
Advanced Training System for NeuroForge

Comprehensive training system with support for LoRA, QLoRA, gradient checkpointing,
mixed precision, and advanced optimization techniques.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from dataclasses import asdict
import wandb

from ..core.config import TrainingConfig
from .optimizer import OptimizerFactory
from .scheduler import SchedulerFactory
from .data_loader import DataLoaderFactory


logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """
    Advanced trainer with support for modern training techniques.
    
    Features:
    - LoRA/QLoRA fine-tuning
    - Gradient checkpointing
    - Mixed precision training
    - Advanced optimizers and schedulers
    - Comprehensive logging and monitoring
    - Checkpointing and resuming
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history: List[Dict[str, Any]] = []
        
        # Setup components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_scaler()
        self._setup_logging()
        
        # Move model to device
        self.model = self.model.to(device)
        
        logger.info(f"Advanced trainer initialized for device: {device}")
        logger.info(f"Model parameters: {self._count_parameters():,}")
    
    def _setup_optimizer(self):
        """Setup optimizer with advanced configurations."""
        self.optimizer = OptimizerFactory.create_optimizer(
            model=self.model,
            config=self.config
        )
        logger.info(f"Optimizer created: {type(self.optimizer).__name__}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        self.scheduler = SchedulerFactory.create_scheduler(
            optimizer=self.optimizer,
            config=self.config
        )
        logger.info(f"Scheduler created: {type(self.scheduler).__name__}")
    
    def _setup_scaler(self):
        """Setup gradient scaler for mixed precision."""
        if self.config.use_fp16 or self.config.use_bf16:
            self.scaler = GradScaler('cuda' if self.device.type == 'cuda' else 'cpu')
        else:
            self.scaler = None
        logger.info(f"Mixed precision: FP16={self.config.use_fp16}, BF16={self.config.use_bf16}")
    
    def _setup_logging(self):
        """Setup logging and monitoring."""
        if self.config.use_wandb:
            wandb.init(
                project="neuroforge-training",
                config=asdict(self.config),
                name=f"training_{int(time.time())}"
            )
            logger.info("Wandb logging initialized")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    async def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting training")
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            await self._load_checkpoint(resume_from_checkpoint)
        
        # Training loop
        start_time = time.time()
        
        try:
            while self.current_step < self.config.max_steps:
                # Training epoch
                train_metrics = await self._train_epoch(train_loader)
                
                # Validation
                val_metrics = None
                if val_loader and self.current_step % self.config.eval_steps == 0:
                    val_metrics = await self._validate(val_loader)
                
                # Logging
                await self._log_metrics(train_metrics, val_metrics)
                
                # Checkpointing
                if self.current_step % self.config.save_steps == 0:
                    await self._save_checkpoint()
                
                # Update step counter
                self.current_step += 1
                
                # Check for early stopping
                if val_metrics and val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    await self._save_checkpoint(is_best=True)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        # Final checkpoint
        await self._save_checkpoint(is_final=True)
        
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'total_steps': self.current_step,
            'total_epochs': self.current_epoch,
            'best_loss': self.best_loss,
            'training_time': training_time,
            'final_metrics': self.training_history[-1] if self.training_history else {}
        }
    
    async def _train_epoch(self, train_loader: DataLoader) -> Dict[str, Any]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    loss, metrics = self._forward_pass(batch)
            else:
                loss, metrics = self._forward_pass(batch)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Collect metrics
            epoch_losses.append(loss.item())
            epoch_metrics.update(metrics)
            
            # Log progress
            if batch_idx % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.current_step}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        self.current_epoch += 1
        
        return {
            'loss': np.mean(epoch_losses),
            'lr': self.optimizer.param_groups[0]['lr'],
            **epoch_metrics
        }
    
    async def _validate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Validate the model."""
        self.model.eval()
        
        val_losses = []
        val_metrics = {}
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        loss, metrics = self._forward_pass(batch)
                else:
                    loss, metrics = self._forward_pass(batch)
                
                val_losses.append(loss.item())
                val_metrics.update(metrics)
        
        self.model.train()
        
        return {
            'loss': np.mean(val_losses),
            **val_metrics
        }
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through the model."""
        # Extract inputs and targets
        input_ids = batch['input_ids']
        targets = batch.get('labels', input_ids)
        
        # Forward pass
        if hasattr(self.model, 'forward'):
            outputs = self.model(input_ids=input_ids, labels=targets)
            
            if isinstance(outputs, tuple):
                logits, loss = outputs
            else:
                logits = outputs
                loss = None
        else:
            raise ValueError("Model must have a forward method")
        
        # Compute loss if not provided
        if loss is None:
            loss = nn.CrossEntropyropy()(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Compute additional metrics
        metrics = {
            'perplexity': torch.exp(loss).item(),
            'accuracy': self._compute_accuracy(logits, targets)
        }
        
        return loss, metrics
    
    def _compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy."""
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float()
        return correct.mean().item()
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()}
    
    async def _log_metrics(self, train_metrics: Dict[str, Any], val_metrics: Optional[Dict[str, Any]]):
        """Log training metrics."""
        # Store in history
        step_metrics = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'train': train_metrics
        }
        
        if val_metrics:
            step_metrics['val'] = val_metrics
        
        self.training_history.append(step_metrics)
        
        # Log to wandb
        if self.config.use_wandb:
            log_dict = {f'train/{k}': v for k, v in train_metrics.items()}
            if val_metrics:
                log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
            wandb.log(log_dict, step=self.current_step)
    
    async def _save_checkpoint(
        self, 
        is_best: bool = False, 
        is_final: bool = False
    ) -> str:
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Determine checkpoint name
        if is_best:
            checkpoint_name = "best_model.pt"
        elif is_final:
            checkpoint_name = "final_model.pt"
        else:
            checkpoint_name = f"checkpoint_step_{self.current_step}.pt"
        
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'config': asdict(self.config),
            'training_history': self.training_history
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    async def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_step = checkpoint.get('current_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resuming from step {self.current_step}, epoch {self.current_epoch}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'total_parameters': self._count_parameters(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_history_length': len(self.training_history)
        }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history.copy()
    
    async def cleanup(self):
        """Cleanup training resources."""
        if self.config.use_wandb:
            wandb.finish()
        logger.info("Training cleanup completed")
