#!/usr/bin/env python3
"""
NeuroForge Demo Server
A simplified version to demonstrate the live project
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="NeuroForge Demo",
    description="Advanced Multi-Modal AI Platform Demo",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Demo data
demo_models = [
    {
        "name": "retnet-default",
        "type": "RetNet",
        "parameters": "1.1B",
        "description": "Revolutionary retention mechanism with O(n) complexity"
    },
    {
        "name": "moe-default", 
        "type": "Mixture of Experts",
        "parameters": "2.3B",
        "description": "Dynamic routing with 8 experts and load balancing"
    },
    {
        "name": "multimodal-default",
        "type": "Multi-Modal",
        "parameters": "3.2B", 
        "description": "Unified text, vision, and audio processing"
    }
]

# Routes
@app.get("/")
async def root():
    return {
        "message": "NeuroForge Advanced AI Platform",
        "version": "1.0.0",
        "status": "online",
        "features": [
            "RetNet Architecture",
            "Mixture of Experts (MoE)",
            "Multi-Modal Processing", 
            "Real-Time Streaming",
            "WebSocket Support",
            "Modern FastAPI Backend"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": "running",
        "models_loaded": len(demo_models)
    }

@app.get("/models")
async def list_models():
    return {"models": demo_models}

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    for model in demo_models:
        if model["name"] == model_name:
            return model
    return {"error": "Model not found"}

@app.post("/generate/text")
async def generate_text(request: Dict[str, Any]):
    prompt = request.get("prompt", "Hello, how are you?")
    model_name = request.get("model_name", "retnet-default")
    max_tokens = request.get("max_tokens", 100)
    
    # Simulate AI response
    responses = {
        "retnet-default": f"RetNet Response: I'm doing well, thank you! I'm powered by the revolutionary RetNet architecture with O(n) complexity. {prompt}",
        "moe-default": f"MoE Response: Great question! As a Mixture of Experts model, I can dynamically route your query to the most appropriate expert. {prompt}",
        "multimodal-default": f"Multi-Modal Response: I can process text, images, and audio! Here's my response to: {prompt}"
    }
    
    generated_text = responses.get(model_name, f"AI Response: {prompt}")
    
    return {
        "generated_text": generated_text,
        "model_name": model_name,
        "tokens_generated": len(generated_text.split()),
        "generation_time": 0.245,
        "timestamp": time.time()
    }

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await manager.send_personal_message(
            json.dumps({
                "type": "welcome",
                "message": "Connected to NeuroForge Chat!",
                "timestamp": time.time()
            }), 
            websocket
        )
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "chat":
                user_message = message.get("message", "")
                
                # Simulate AI thinking
                await manager.send_personal_message(
                    json.dumps({
                        "type": "typing",
                        "message": "AI is thinking...",
                        "timestamp": time.time()
                    }), 
                    websocket
                )
                
                await asyncio.sleep(1)  # Simulate processing time
                
                # Generate response
                ai_response = f"NeuroForge AI: I understand you said '{user_message}'. I'm powered by cutting-edge RetNet and MoE architectures. How can I help you further?"
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "chat_response",
                        "message": ai_response,
                        "timestamp": time.time()
                    }), 
                    websocket
                )
            
            elif message.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }), 
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/demo")
async def demo_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NeuroForge Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 40px; }
            .header h1 { color: #2563eb; margin-bottom: 10px; }
            .header p { color: #666; font-size: 18px; }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
            .feature { background: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #2563eb; }
            .feature h3 { color: #1e40af; margin-top: 0; }
            .chat-demo { margin-top: 40px; }
            .chat-container { border: 1px solid #ddd; border-radius: 8px; height: 400px; overflow-y: auto; padding: 20px; background: #fafafa; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user-message { background: #dbeafe; text-align: right; }
            .ai-message { background: #f0f9ff; }
            .input-container { display: flex; margin-top: 20px; }
            .input-container input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px 0 0 5px; }
            .input-container button { padding: 10px 20px; background: #2563eb; color: white; border: none; border-radius: 0 5px 5px 0; cursor: pointer; }
            .status { text-align: center; margin: 20px 0; padding: 10px; background: #dcfce7; color: #166534; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 NeuroForge Advanced AI Platform</h1>
                <p>Revolutionary AI with RetNet, MoE, and Multi-Modal Processing</p>
            </div>
            
            <div class="status">
                ✅ System Online | 🔗 WebSocket Connected | 🧠 AI Models Ready
            </div>
            
            <div class="features">
                <div class="feature">
                    <h3>🧠 RetNet Architecture</h3>
                    <p>Revolutionary retention mechanism with O(n) complexity instead of O(n²) for attention. Enables efficient long-sequence modeling.</p>
                </div>
                <div class="feature">
                    <h3>⚡ Mixture of Experts (MoE)</h3>
                    <p>Dynamic routing with load balancing across 8 experts. Scales model capacity without proportional computation increase.</p>
                </div>
                <div class="feature">
                    <h3>🎯 Multi-Modal Processing</h3>
                    <p>Unified processing of text, vision, and audio with advanced fusion mechanisms and cross-modal attention.</p>
                </div>
                <div class="feature">
                    <h3>🌐 Real-Time Streaming</h3>
                    <p>WebSocket-based streaming inference with <100ms latency for interactive AI experiences.</p>
                </div>
                <div class="feature">
                    <h3>🏗️ Modern Architecture</h3>
                    <p>FastAPI backend, Next.js frontend, Docker containerization, and Kubernetes deployment ready.</p>
                </div>
                <div class="feature">
                    <h3>📊 Production Ready</h3>
                    <p>Comprehensive monitoring with Prometheus, Grafana, authentication, rate limiting, and enterprise security.</p>
                </div>
            </div>
            
            <div class="chat-demo">
                <h2>💬 Live Chat Demo</h2>
                <div id="chatContainer" class="chat-container">
                    <div class="message ai-message">
                        <strong>NeuroForge AI:</strong> Hello! I'm your advanced AI assistant powered by cutting-edge RetNet and MoE architectures. How can I help you today?
                    </div>
                </div>
                <div class="input-container">
                    <input type="text" id="messageInput" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket('ws://localhost:8000/ws/chat');
            const chatContainer = document.getElementById('chatContainer');
            const messageInput = document.getElementById('messageInput');
            
            ws.onopen = function(event) {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'chat_response') {
                    addMessage(data.message, 'ai');
                } else if (data.type === 'typing') {
                    addMessage(data.message, 'ai', true);
                }
            };
            
            function addMessage(message, sender, isTyping = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                if (isTyping) {
                    messageDiv.innerHTML = `<em>${message}</em>`;
                } else {
                    messageDiv.innerHTML = `<strong>NeuroForge AI:</strong> ${message}`;
                }
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function sendMessage() {
                const message = messageInput.value.trim();
                if (message) {
                    addMessage(message, 'user');
                    ws.send(JSON.stringify({
                        type: 'chat',
                        message: message
                    }));
                    messageInput.value = '';
                }
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("🚀 Starting NeuroForge Demo Server...")
    print("📡 API will be available at: http://localhost:8000")
    print("🌐 Demo page will be available at: http://localhost:8000/demo")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("💬 WebSocket chat at: ws://localhost:8000/ws/chat")
    print("\n✨ Features:")
    print("   • RetNet Architecture Demo")
    print("   • Mixture of Experts (MoE)")
    print("   • Multi-Modal Processing")
    print("   • Real-Time WebSocket Chat")
    print("   • Interactive Demo Page")
    print("\n🎯 Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
