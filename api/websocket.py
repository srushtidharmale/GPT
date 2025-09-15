"""
WebSocket Router for NeuroForge

Real-time WebSocket endpoints for streaming inference, live chat,
and interactive AI experiences.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect, Depends
from fastapi.routing import APIRouter
import structlog

from neuroforge.core.engine import NeuroForgeEngine
from .server import get_engine

logger = structlog.get_logger(__name__)

# Create WebSocket router
websocket_router = APIRouter()

# Connection manager
class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_info[connection_id] = {
            'connected_at': time.time(),
            'last_activity': time.time(),
            'requests_processed': 0
        }
        logger.info("WebSocket connection established", connection_id=connection_id)
    
    def disconnect(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.connection_info:
            del self.connection_info[connection_id]
        logger.info("WebSocket connection closed", connection_id=connection_id)
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error("Failed to send message", connection_id=connection_id, error=str(e))
                self.disconnect(connection_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections."""
        for connection_id in list(self.active_connections.keys()):
            await self.send_message(connection_id, message)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            connection_id: {
                **info,
                'connected_duration': time.time() - info['connected_at']
            }
            for connection_id, info in self.connection_info.items()
        }


# Global connection manager
manager = ConnectionManager()


@websocket_router.websocket("/chat")
async def websocket_chat(
    websocket: WebSocket,
    engine: NeuroForgeEngine = Depends(get_engine)
):
    """WebSocket endpoint for real-time chat."""
    connection_id = str(uuid.uuid4())
    
    try:
        await manager.connect(websocket, connection_id)
        
        # Send welcome message
        await manager.send_message(connection_id, {
            'type': 'welcome',
            'message': 'Connected to NeuroForge chat',
            'connection_id': connection_id,
            'timestamp': time.time()
        })
        
        # Handle messages
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Update connection info
                manager.connection_info[connection_id]['last_activity'] = time.time()
                manager.connection_info[connection_id]['requests_processed'] += 1
                
                # Process message
                await handle_chat_message(connection_id, message, engine)
                
            except json.JSONDecodeError:
                await manager.send_message(connection_id, {
                    'type': 'error',
                    'message': 'Invalid JSON message',
                    'timestamp': time.time()
                })
            except Exception as e:
                logger.error("Error processing chat message", error=str(e))
                await manager.send_message(connection_id, {
                    'type': 'error',
                    'message': f'Error: {str(e)}',
                    'timestamp': time.time()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error("WebSocket chat error", error=str(e))
        manager.disconnect(connection_id)


@websocket_router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    engine: NeuroForgeEngine = Depends(get_engine)
):
    """WebSocket endpoint for streaming inference."""
    connection_id = str(uuid.uuid4())
    
    try:
        await manager.connect(websocket, connection_id)
        
        # Send welcome message
        await manager.send_message(connection_id, {
            'type': 'stream_ready',
            'message': 'Streaming inference ready',
            'connection_id': connection_id,
            'timestamp': time.time()
        })
        
        # Handle streaming requests
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Update connection info
                manager.connection_info[connection_id]['last_activity'] = time.time()
                manager.connection_info[connection_id]['requests_processed'] += 1
                
                # Process streaming request
                await handle_streaming_request(connection_id, message, engine)
                
            except json.JSONDecodeError:
                await manager.send_message(connection_id, {
                    'type': 'error',
                    'message': 'Invalid JSON message',
                    'timestamp': time.time()
                })
            except Exception as e:
                logger.error("Error processing streaming request", error=str(e))
                await manager.send_message(connection_id, {
                    'type': 'error',
                    'message': f'Error: {str(e)}',
                    'timestamp': time.time()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error("WebSocket stream error", error=str(e))
        manager.disconnect(connection_id)


@websocket_router.websocket("/multimodal")
async def websocket_multimodal(
    websocket: WebSocket,
    engine: NeuroForgeEngine = Depends(get_engine)
):
    """WebSocket endpoint for multi-modal processing."""
    connection_id = str(uuid.uuid4())
    
    try:
        await manager.connect(websocket, connection_id)
        
        # Send welcome message
        await manager.send_message(connection_id, {
            'type': 'multimodal_ready',
            'message': 'Multi-modal processing ready',
            'connection_id': connection_id,
            'timestamp': time.time()
        })
        
        # Handle multi-modal requests
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Update connection info
                manager.connection_info[connection_id]['last_activity'] = time.time()
                manager.connection_info[connection_id]['requests_processed'] += 1
                
                # Process multi-modal request
                await handle_multimodal_request(connection_id, message, engine)
                
            except json.JSONDecodeError:
                await manager.send_message(connection_id, {
                    'type': 'error',
                    'message': 'Invalid JSON message',
                    'timestamp': time.time()
                })
            except Exception as e:
                logger.error("Error processing multi-modal request", error=str(e))
                await manager.send_message(connection_id, {
                    'type': 'error',
                    'message': f'Error: {str(e)}',
                    'timestamp': time.time()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error("WebSocket multimodal error", error=str(e))
        manager.disconnect(connection_id)


async def handle_chat_message(
    connection_id: str,
    message: Dict[str, Any],
    engine: NeuroForgeEngine
):
    """Handle chat message."""
    message_type = message.get('type')
    
    if message_type == 'chat':
        # Handle chat message
        user_message = message.get('message', '')
        model_name = message.get('model_name', 'retnet-default')
        
        # Send typing indicator
        await manager.send_message(connection_id, {
            'type': 'typing',
            'message': 'AI is thinking...',
            'timestamp': time.time()
        })
        
        # Generate response
        try:
            response = await engine.generate_text(
                model_name=model_name,
                prompt=user_message,
                max_tokens=100,
                temperature=0.8,
                stream=False
            )
            
            # Send response
            await manager.send_message(connection_id, {
                'type': 'chat_response',
                'message': response,
                'model_name': model_name,
                'timestamp': time.time()
            })
        
        except Exception as e:
            await manager.send_message(connection_id, {
                'type': 'error',
                'message': f'Failed to generate response: {str(e)}',
                'timestamp': time.time()
            })
    
    elif message_type == 'ping':
        # Handle ping
        await manager.send_message(connection_id, {
            'type': 'pong',
            'timestamp': time.time()
        })
    
    else:
        await manager.send_message(connection_id, {
            'type': 'error',
            'message': f'Unknown message type: {message_type}',
            'timestamp': time.time()
        })


async def handle_streaming_request(
    connection_id: str,
    message: Dict[str, Any],
    engine: NeuroForgeEngine
):
    """Handle streaming inference request."""
    message_type = message.get('type')
    
    if message_type == 'generate':
        # Handle text generation
        prompt = message.get('prompt', '')
        model_name = message.get('model_name', 'retnet-default')
        max_tokens = message.get('max_tokens', 100)
        temperature = message.get('temperature', 0.8)
        
        # Send start message
        await manager.send_message(connection_id, {
            'type': 'generation_start',
            'message': 'Starting generation...',
            'timestamp': time.time()
        })
        
        # Stream generation
        try:
            generated_text = ""
            async for token in engine.generate_text(
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            ):
                generated_text += token
                
                # Send token
                await manager.send_message(connection_id, {
                    'type': 'token',
                    'token': token,
                    'generated_text': generated_text,
                    'timestamp': time.time()
                })
            
            # Send completion message
            await manager.send_message(connection_id, {
                'type': 'generation_complete',
                'generated_text': generated_text,
                'total_tokens': len(generated_text.split()),
                'timestamp': time.time()
            })
        
        except Exception as e:
            await manager.send_message(connection_id, {
                'type': 'error',
                'message': f'Generation failed: {str(e)}',
                'timestamp': time.time()
            })
    
    elif message_type == 'ping':
        # Handle ping
        await manager.send_message(connection_id, {
            'type': 'pong',
            'timestamp': time.time()
        })
    
    else:
        await manager.send_message(connection_id, {
            'type': 'error',
            'message': f'Unknown message type: {message_type}',
            'timestamp': time.time()
        })


async def handle_multimodal_request(
    connection_id: str,
    message: Dict[str, Any],
    engine: NeuroForgeEngine
):
    """Handle multi-modal processing request."""
    message_type = message.get('type')
    
    if message_type == 'process':
        # Handle multi-modal processing
        text = message.get('text')
        model_name = message.get('model_name', 'multimodal-default')
        
        # Send start message
        await manager.send_message(connection_id, {
            'type': 'processing_start',
            'message': 'Starting multi-modal processing...',
            'timestamp': time.time()
        })
        
        # Process multi-modal input
        try:
            result = await engine.process_multimodal(
                model_name=model_name,
                text=text
            )
            
            # Send result
            await manager.send_message(connection_id, {
                'type': 'processing_complete',
                'result': result,
                'timestamp': time.time()
            })
        
        except Exception as e:
            await manager.send_message(connection_id, {
                'type': 'error',
                'message': f'Processing failed: {str(e)}',
                'timestamp': time.time()
            })
    
    elif message_type == 'ping':
        # Handle ping
        await manager.send_message(connection_id, {
            'type': 'pong',
            'timestamp': time.time()
        })
    
    else:
        await manager.send_message(connection_id, {
            'type': 'error',
            'message': f'Unknown message type: {message_type}',
            'timestamp': time.time()
        })


# Utility endpoints
@websocket_router.get("/connections")
async def get_connections():
    """Get WebSocket connection information."""
    return {
        'active_connections': len(manager.active_connections),
        'connections': manager.get_connection_info()
    }


@websocket_router.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all WebSocket connections."""
    await manager.broadcast(message)
    return {"message": "Broadcast sent", "connections": len(manager.active_connections)}
