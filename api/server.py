"""
FastAPI Server for NeuroForge

Modern REST API and WebSocket server with advanced features including
authentication, rate limiting, monitoring, and real-time streaming.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog

from neuroforge.core.engine import NeuroForgeEngine
from neuroforge.core.config import NeuroForgeConfig
from .routes import router
from .websocket import websocket_router
from .middleware import setup_middleware
from .auth import verify_token


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('neuroforge_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('neuroforge_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('neuroforge_active_connections', 'Active WebSocket connections')
MODEL_INFERENCE_TIME = Histogram('neuroforge_model_inference_seconds', 'Model inference time', ['model_name'])

# Global engine instance
engine: Optional[NeuroForgeEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global engine
    
    # Startup
    logger.info("Starting NeuroForge API server")
    
    try:
        # Load configuration
        config = NeuroForgeConfig()
        
        # Initialize engine
        engine = NeuroForgeEngine(config)
        
        # Create default models if they don't exist
        await create_default_models()
        
        logger.info("NeuroForge API server started successfully")
        
    except Exception as e:
        logger.error("Failed to start NeuroForge API server", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down NeuroForge API server")
    
    if engine:
        # Cleanup engine resources
        pass
    
    logger.info("NeuroForge API server shutdown complete")


async def create_default_models():
    """Create default models if they don't exist."""
    global engine
    
    if not engine:
        return
    
    # Create RetNet model if it doesn't exist
    if 'retnet-default' not in engine.list_models():
        try:
            engine.create_model(
                model_name='retnet-default',
                model_type='retnet',
                vocab_size=50257,
                embed_dim=1536,
                num_heads=12,
                num_layers=24,
                max_seq_len=2048
            )
            logger.info("Created default RetNet model")
        except Exception as e:
            logger.warning("Failed to create default RetNet model", error=str(e))
    
    # Create MoE model if it doesn't exist
    if 'moe-default' not in engine.list_models():
        try:
            engine.create_model(
                model_name='moe-default',
                model_type='moe',
                vocab_size=50257,
                embed_dim=1536,
                num_heads=12,
                num_layers=24,
                num_experts=8,
                top_k=2
            )
            logger.info("Created default MoE model")
        except Exception as e:
            logger.warning("Failed to create default MoE model", error=str(e))
    
    # Create multi-modal model if it doesn't exist
    if 'multimodal-default' not in engine.list_models():
        try:
            engine.create_model(
                model_name='multimodal-default',
                model_type='multimodal',
                vocab_size=50257,
                text_embed_dim=1536,
                text_num_heads=12,
                text_num_layers=24,
                vision_embed_dim=512,
                audio_embed_dim=256,
                fusion_dim=512
            )
            logger.info("Created default multi-modal model")
        except Exception as e:
            logger.warning("Failed to create default multi-modal model", error=str(e))


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="NeuroForge API",
        description="Advanced Multi-Modal AI Platform API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/ws")
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        global engine
        
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        
        try:
            health_status = await engine.health_check()
            return health_status
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type="text/plain")
    
    # Add system info endpoint
    @app.get("/system")
    async def system_info():
        """System information endpoint."""
        global engine
        
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        
        return engine.get_system_info()
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Update metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Log request
        logger.info(
            "Request processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            client_ip=request.client.host if request.client else None
        )
        
        return response
    
    return app


def get_engine() -> NeuroForgeEngine:
    """Get the global engine instance."""
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return engine


# Security
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    # In a real implementation, you would verify the JWT token
    # For now, we'll just return a placeholder user
    token = credentials.credentials
    
    # Verify token (placeholder)
    user = await verify_token(token)
    
    return user


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api.server:create_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
