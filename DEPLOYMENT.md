# NeuroForge Deployment Guide

This guide covers deploying NeuroForge in various environments, from local development to production cloud deployments.

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**
```bash
git clone <repository-url>
cd neuroforge
./scripts/setup.sh
```

2. **Access the Platform**
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Grafana: http://localhost:3001 (admin/admin)

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   Mobile App    │    │   API Clients   │
│   (Next.js)     │    │   (React Native)│    │   (Python/JS)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      API Gateway          │
                    │   (FastAPI + WebSocket)   │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────┴─────────┐  ┌─────────┴─────────┐  ┌─────────┴─────────┐
│  Text Processing  │  │  Vision Pipeline  │  │  Audio Pipeline  │
│   (NeuroForge)    │  │   (VisionForge)   │  │   (AudioForge)   │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │   Vector Database         │
                    │   (Pinecone/Weaviate)     │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │   Model Storage           │
                    │   (HuggingFace Hub)       │
                    └────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
NEUROFORGE_ENV=production
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Database Configuration
POSTGRES_URL=postgresql://user:password@host:5432/database
REDIS_URL=redis://host:6379

# Model Configuration
MODEL_CACHE_DIR=/app/models
DATA_DIR=/app/data
CACHE_DIR=/app/cache
OUTPUT_DIR=/app/outputs

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
WANDB_ENABLED=false
WANDB_PROJECT=neuroforge-production

# Security
JWT_SECRET=your-secret-key
CORS_ORIGINS=https://yourdomain.com
```

### Model Configuration

Create `config.yaml`:

```yaml
model:
  name: "neuroforge-production"
  type: "retnet"
  vocab_size: 50257
  embed_dim: 1536
  num_heads: 12
  num_layers: 24
  max_seq_len: 2048
  dropout: 0.1
  use_flash_attn: true
  use_gradient_checkpointing: true

training:
  batch_size: 32
  learning_rate: 6e-4
  weight_decay: 0.01
  warmup_steps: 1000
  max_steps: 100000
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  use_lora: true
  lora_rank: 16
  lora_alpha: 32.0
  use_fp16: true

inference:
  max_new_tokens: 512
  temperature: 0.8
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  stream_chunk_size: 10
  stream_delay: 0.05

system:
  device: "auto"
  num_gpus: 1
  distributed: false
  mixed_precision: true
  max_memory_usage: 0.9
  log_level: "INFO"
```

## ☁️ Cloud Deployment

### AWS Deployment

1. **EC2 Instance Setup**
```bash
# Launch EC2 instance (g4dn.xlarge or larger)
# Install Docker and Docker Compose
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

2. **Deploy with Docker Compose**
```bash
# Clone repository
git clone <repository-url>
cd neuroforge

# Configure environment
cp .env.example .env
# Edit .env with your configuration

# Deploy
docker-compose up -d
```

3. **Configure Load Balancer**
```yaml
# nginx/nginx.conf for production
upstream neuroforge_api {
    server neuroforge-api:8000;
    keepalive 32;
}

upstream neuroforge_frontend {
    server neuroforge-frontend:3000;
    keepalive 32;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # SSL redirect
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # API routes
    location /api/ {
        proxy_pass http://neuroforge_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket routes
    location /ws/ {
        proxy_pass http://neuroforge_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Frontend routes
    location / {
        proxy_pass http://neuroforge_frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Kubernetes Deployment

1. **Create Namespace**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: neuroforge
```

2. **Deploy API Service**
```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuroforge-api
  namespace: neuroforge
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neuroforge-api
  template:
    metadata:
      labels:
        app: neuroforge-api
    spec:
      containers:
      - name: api
        image: neuroforge-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: NEUROFORGE_ENV
          value: "production"
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: neuroforge-secrets
              key: postgres-url
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: neuroforge-api
  namespace: neuroforge
spec:
  selector:
    app: neuroforge-api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

3. **Deploy Frontend**
```yaml
# k8s/frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuroforge-frontend
  namespace: neuroforge
spec:
  replicas: 2
  selector:
    matchLabels:
      app: neuroforge-frontend
  template:
    metadata:
      labels:
        app: neuroforge-frontend
    spec:
      containers:
      - name: frontend
        image: neuroforge-frontend:latest
        ports:
        - containerPort: 3000
        env:
        - name: NEXT_PUBLIC_API_URL
          value: "http://neuroforge-api:8000"
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.5"
          limits:
            memory: "1Gi"
            cpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: neuroforge-frontend
  namespace: neuroforge
spec:
  selector:
    app: neuroforge-frontend
  ports:
  - port: 3000
    targetPort: 3000
  type: ClusterIP
```

4. **Deploy with Ingress**
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: neuroforge-ingress
  namespace: neuroforge
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: neuroforge-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: neuroforge-api
            port:
              number: 8000
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: neuroforge-api
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: neuroforge-frontend
            port:
              number: 3000
```

## 📊 Monitoring & Observability

### Prometheus Metrics

NeuroForge exposes comprehensive metrics:

- `neuroforge_requests_total`: Total API requests
- `neuroforge_request_duration_seconds`: Request duration
- `neuroforge_active_connections`: Active WebSocket connections
- `neuroforge_model_inference_seconds`: Model inference time
- `neuroforge_training_loss`: Training loss
- `neuroforge_model_parameters`: Model parameter count

### Grafana Dashboards

Pre-configured dashboards for:
- API Performance
- Model Metrics
- System Resources
- Training Progress
- User Activity

### Logging

Structured logging with:
- Request/response logging
- Error tracking
- Performance metrics
- Audit trails

## 🔒 Security

### Authentication & Authorization

- JWT-based authentication
- Role-based access control
- API key management
- Rate limiting

### Data Protection

- Encryption at rest and in transit
- Secure model storage
- Data anonymization
- GDPR compliance

### Network Security

- HTTPS/TLS encryption
- CORS configuration
- Security headers
- DDoS protection

## 🚀 Performance Optimization

### Scaling Strategies

1. **Horizontal Scaling**
   - Multiple API instances
   - Load balancing
   - Auto-scaling groups

2. **Vertical Scaling**
   - GPU optimization
   - Memory management
   - CPU optimization

3. **Caching**
   - Redis caching
   - Model caching
   - Response caching

### Performance Tuning

```yaml
# Production optimizations
model:
  use_flash_attn: true
  use_gradient_checkpointing: true
  mixed_precision: true

inference:
  use_kv_cache: true
  kv_cache_size: 1024
  stream_chunk_size: 20

system:
  max_memory_usage: 0.9
  num_workers: 4
  prefetch_factor: 2
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision

2. **Slow Inference**
   - Enable KV caching
   - Optimize model size
   - Use GPU acceleration

3. **Connection Issues**
   - Check firewall settings
   - Verify network configuration
   - Monitor connection limits

### Debugging

```bash
# View logs
docker-compose logs -f neuroforge-api

# Check service health
curl http://localhost:8000/health

# Monitor resources
docker stats

# Access container shell
docker-compose exec neuroforge-api bash
```

## 📈 Scaling to Production

### High Availability

- Multi-region deployment
- Database replication
- Load balancer health checks
- Automatic failover

### Backup & Recovery

- Automated backups
- Point-in-time recovery
- Model versioning
- Configuration management

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push images
      run: |
        docker build -t neuroforge-api:latest .
        docker push neuroforge-api:latest
    
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/
        kubectl rollout restart deployment/neuroforge-api
        kubectl rollout restart deployment/neuroforge-frontend
```

## 📚 Additional Resources

- [API Documentation](http://localhost:8000/docs)
- [Model Architecture Guide](./ARCHITECTURE.md)
- [Training Guide](./TRAINING.md)
- [Contributing Guide](./CONTRIBUTING.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

---

**NeuroForge** - Deploying the future of AI, one innovation at a time.
