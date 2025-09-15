"""
API Routes for NeuroForge

REST API endpoints for model management, inference, training, and system operations.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch
import json

from neuroforge.core.engine import NeuroForgeEngine
from neuroforge.core.config import NeuroForgeConfig
from .server import get_engine, get_current_user


# Request/Response models
class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text generation")
    model_name: str = Field(default="retnet-default", description="Model to use for generation")
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Top-k sampling")
    top_p: float = Field(default=1.0, ge=0.1, le=1.0, description="Top-p sampling")
    stream: bool = Field(default=False, description="Whether to stream the output")
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0, description="Repetition penalty")


class TextGenerationResponse(BaseModel):
    generated_text: str
    model_name: str
    tokens_generated: int
    generation_time: float
    timestamp: float


class MultiModalRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Text input")
    model_name: str = Field(default="multimodal-default", description="Model to use")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum tokens to generate")


class MultiModalResponse(BaseModel):
    result: Dict[str, Any]
    model_name: str
    processing_time: float
    timestamp: float


class ModelInfo(BaseModel):
    name: str
    type: str
    parameters: int
    device: str
    config: Dict[str, Any]


class ModelCreateRequest(BaseModel):
    model_name: str = Field(..., description="Name for the new model")
    model_type: str = Field(..., description="Type of model to create")
    config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")


class TrainingRequest(BaseModel):
    model_name: str = Field(..., description="Model to train")
    dataset_name: str = Field(default="fineweb-edu", description="Dataset to use")
    max_steps: int = Field(default=1000, ge=1, description="Maximum training steps")
    learning_rate: float = Field(default=6e-4, ge=1e-6, le=1e-2, description="Learning rate")
    batch_size: int = Field(default=32, ge=1, le=128, description="Batch size")


class TrainingResponse(BaseModel):
    training_id: str
    model_name: str
    status: str
    message: str


# Create router
router = APIRouter()


# Model Management Endpoints
@router.get("/models", response_model=List[ModelInfo])
async def list_models(engine: NeuroForgeEngine = Depends(get_engine)):
    """List all available models."""
    try:
        models = []
        for model_name in engine.list_models():
            model_info = engine.get_model_info(model_name)
            models.append(ModelInfo(**model_info))
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(
    model_name: str,
    engine: NeuroForgeEngine = Depends(get_engine)
):
    """Get information about a specific model."""
    try:
        model_info = engine.get_model_info(model_name)
        return ModelInfo(**model_info)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/models", response_model=ModelInfo)
async def create_model(
    request: ModelCreateRequest,
    engine: NeuroForgeEngine = Depends(get_engine),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new model."""
    try:
        model = engine.create_model(
            model_name=request.model_name,
            model_type=request.model_type,
            **request.config
        )
        model_info = engine.get_model_info(request.model_name)
        return ModelInfo(**model_info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    engine: NeuroForgeEngine = Depends(get_engine),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a model."""
    try:
        engine.delete_model(model_name)
        return {"message": f"Model {model_name} deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.post("/models/{model_name}/save")
async def save_model(
    model_name: str,
    engine: NeuroForgeEngine = Depends(get_engine),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Save a model to disk."""
    try:
        save_path = engine.save_model(model_name)
        return {"message": f"Model {model_name} saved successfully", "path": save_path}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")


# Text Generation Endpoints
@router.post("/generate/text", response_model=TextGenerationResponse)
async def generate_text(
    request: TextGenerationRequest,
    engine: NeuroForgeEngine = Depends(get_engine)
):
    """Generate text using a model."""
    start_time = time.time()
    
    try:
        if request.stream:
            # Stream generation
            generated_text = ""
            async for token in engine.generate_text(
                model_name=request.model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stream=True
            ):
                generated_text += token
        
        else:
            # Batch generation
            generated_text = await engine.generate_text(
                model_name=request.model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stream=False
            )
        
        generation_time = time.time() - start_time
        
        return TextGenerationResponse(
            generated_text=generated_text,
            model_name=request.model_name,
            tokens_generated=len(generated_text.split()),
            generation_time=generation_time,
            timestamp=time.time()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate text: {str(e)}")


@router.post("/generate/text/stream")
async def generate_text_stream(
    request: TextGenerationRequest,
    engine: NeuroForgeEngine = Depends(get_engine)
):
    """Stream text generation."""
    try:
        async def generate():
            async for token in engine.generate_text(
                model_name=request.model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stream=True
            ):
                yield f"data: {json.dumps({'token': token, 'timestamp': time.time()})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream text: {str(e)}")


# Multi-Modal Endpoints
@router.post("/generate/multimodal", response_model=MultiModalResponse)
async def generate_multimodal(
    request: MultiModalRequest,
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    engine: NeuroForgeEngine = Depends(get_engine)
):
    """Generate text from multi-modal inputs."""
    start_time = time.time()
    
    try:
        # Process uploaded files
        image_tensor = None
        audio_tensor = None
        
        if image:
            # Process image (placeholder)
            image_tensor = torch.randn(1, 3, 224, 224)  # Placeholder
        
        if audio:
            # Process audio (placeholder)
            audio_tensor = torch.randn(1, 80, 1000)  # Placeholder
        
        # Generate from multi-modal inputs
        result = await engine.process_multimodal(
            model_name=request.model_name,
            text=request.text,
            image=image_tensor,
            audio=audio_tensor
        )
        
        processing_time = time.time() - start_time
        
        return MultiModalResponse(
            result=result,
            model_name=request.model_name,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process multi-modal input: {str(e)}")


# Training Endpoints
@router.post("/training/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    engine: NeuroForgeEngine = Depends(get_engine),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Start training a model."""
    try:
        # Generate training ID
        training_id = f"training_{int(time.time())}"
        
        # Start training in background
        background_tasks.add_task(
            train_model_background,
            engine,
            request.model_name,
            request.dataset_name,
            request.max_steps,
            request.learning_rate,
            request.batch_size,
            training_id
        )
        
        return TrainingResponse(
            training_id=training_id,
            model_name=request.model_name,
            status="started",
            message=f"Training started with ID {training_id}"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


async def train_model_background(
    engine: NeuroForgeEngine,
    model_name: str,
    dataset_name: str,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    training_id: str
):
    """Background training task."""
    try:
        # Setup trainer
        trainer = engine.setup_trainer(model_name)
        
        # Start training (placeholder)
        # In a real implementation, you would call trainer.train()
        await asyncio.sleep(1)  # Simulate training
        
        # Save model after training
        engine.save_model(model_name)
        
    except Exception as e:
        # Log training error
        print(f"Training failed: {e}")


# System Endpoints
@router.get("/system/info")
async def get_system_info(engine: NeuroForgeEngine = Depends(get_engine)):
    """Get system information."""
    try:
        return engine.get_system_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")


@router.get("/system/health")
async def health_check(engine: NeuroForgeEngine = Depends(get_engine)):
    """Perform health check."""
    try:
        return await engine.health_check()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Utility Endpoints
@router.post("/tokenize")
async def tokenize_text(
    text: str,
    engine: NeuroForgeEngine = Depends(get_engine)
):
    """Tokenize text using the default tokenizer."""
    try:
        # Placeholder tokenization
        tokens = text.split()
        token_ids = [hash(token) % 1000 for token in tokens]  # Placeholder
        
        return {
            "text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "vocab_size": 1000  # Placeholder
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to tokenize text: {str(e)}")


@router.post("/detokenize")
async def detokenize_text(
    token_ids: List[int],
    engine: NeuroForgeEngine = Depends(get_engine)
):
    """Detokenize token IDs back to text."""
    try:
        # Placeholder detokenization
        tokens = [f"token_{id}" for id in token_ids]
        text = " ".join(tokens)
        
        return {
            "token_ids": token_ids,
            "tokens": tokens,
            "text": text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detokenize: {str(e)}")
