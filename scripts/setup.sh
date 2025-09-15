#!/bin/bash

# NeuroForge Setup Script
# This script sets up the complete NeuroForge environment

set -e

echo "🚀 Setting up NeuroForge Advanced AI Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS"

# Check for required tools
check_requirements() {
    print_status "Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3.11+ is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js 18+ is required but not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    print_success "All requirements satisfied"
}

# Setup Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install NeuroForge
    pip install -e .
    
    print_success "Python environment setup complete"
}

# Setup Frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd frontend
    
    # Install dependencies
    npm install
    
    # Build frontend
    npm run build
    
    cd ..
    
    print_success "Frontend setup complete"
}

# Setup Docker environment
setup_docker() {
    print_status "Setting up Docker environment..."
    
    # Create necessary directories
    mkdir -p models data cache outputs logs
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    mkdir -p nginx/ssl
    
    # Create default configuration files
    create_config_files
    
    print_success "Docker environment setup complete"
}

# Create configuration files
create_config_files() {
    print_status "Creating configuration files..."
    
    # Create Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'neuroforge-api'
    static_configs:
      - targets: ['neuroforge-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

    # Create Grafana datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Create Nginx configuration
    cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream neuroforge_api {
        server neuroforge-api:8000;
    }

    upstream neuroforge_frontend {
        server neuroforge-frontend:3000;
    }

    server {
        listen 80;
        server_name localhost;

        location /api/ {
            proxy_pass http://neuroforge_api;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        location /ws/ {
            proxy_pass http://neuroforge_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        location / {
            proxy_pass http://neuroforge_frontend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF

    # Create database initialization script
    cat > init.sql << EOF
-- NeuroForge Database Initialization

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models table
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    type VARCHAR(100) NOT NULL,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id),
    status VARCHAR(50) DEFAULT 'pending',
    config JSONB,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inference requests table
CREATE TABLE IF NOT EXISTS inference_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id),
    user_id UUID REFERENCES users(id),
    request_data JSONB,
    response_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_inference_requests_created_at ON inference_requests(created_at);
EOF

    print_success "Configuration files created"
}

# Start services
start_services() {
    print_status "Starting NeuroForge services..."
    
    # Start with Docker Compose
    docker-compose up -d
    
    print_success "Services started successfully"
    print_status "Waiting for services to be ready..."
    
    # Wait for services to be ready
    sleep 30
    
    # Check service health
    check_service_health
}

# Check service health
check_service_health() {
    print_status "Checking service health..."
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API service is healthy"
    else
        print_warning "API service may not be ready yet"
    fi
    
    # Check Frontend
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        print_success "Frontend service is healthy"
    else
        print_warning "Frontend service may not be ready yet"
    fi
    
    # Check Grafana
    if curl -f http://localhost:3001 > /dev/null 2>&1; then
        print_success "Grafana service is healthy"
    else
        print_warning "Grafana service may not be ready yet"
    fi
}

# Display access information
show_access_info() {
    print_success "NeuroForge setup complete!"
    echo ""
    echo "🌐 Access URLs:"
    echo "   Frontend:     http://localhost:3000"
    echo "   API:          http://localhost:8000"
    echo "   API Docs:     http://localhost:8000/docs"
    echo "   Grafana:      http://localhost:3001 (admin/admin)"
    echo "   Prometheus:   http://localhost:9090"
    echo ""
    echo "📚 Quick Start:"
    echo "   1. Open http://localhost:3000 in your browser"
    echo "   2. Start chatting with the AI models"
    echo "   3. Explore the model dashboard"
    echo "   4. Monitor system metrics in Grafana"
    echo ""
    echo "🛠️  Management Commands:"
    echo "   View logs:    docker-compose logs -f"
    echo "   Stop services: docker-compose down"
    echo "   Restart:      docker-compose restart"
    echo ""
}

# Main setup function
main() {
    echo "🎯 NeuroForge Advanced AI Platform Setup"
    echo "========================================"
    echo ""
    
    check_requirements
    setup_python
    setup_frontend
    setup_docker
    start_services
    show_access_info
}

# Run main function
main "$@"
