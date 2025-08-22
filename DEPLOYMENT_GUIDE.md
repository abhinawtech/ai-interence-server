# 🚀 AI Inference Server - Complete Deployment Guide

## Overview

This guide provides comprehensive deployment instructions for the AI Inference Server across different environments: Docker, Kubernetes, and cloud platforms.

## 📋 Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 50GB+ available space
- **GPU**: Optional (NVIDIA with CUDA 12.2+ for GPU acceleration)

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployment)
- kubectl configured with cluster access

## 🐳 Option 1: Docker Deployment (Recommended for Development)

### Quick Start

1. **Build the Application**
   ```bash
   docker build -t ai-inference-server .
   ```

2. **Start with Docker Compose**
   ```bash
   # Basic deployment (AI server + Qdrant)
   docker-compose up -d
   
   # With monitoring
   docker-compose --profile monitoring up -d
   
   # Full production stack
   docker-compose --profile production --profile monitoring up -d
   ```

3. **Verify Deployment**
   ```bash
   curl http://localhost:3000/health
   ```

### Advanced Docker Configuration

#### GPU Support
```bash
# Build with CUDA support
docker build --build-arg FEATURES=cuda -t ai-inference-server:cuda .

# Run with GPU access
docker run --gpus all -p 3000:3000 ai-inference-server:cuda
```

#### Custom Configuration
```bash
# Create data directories
mkdir -p data/{models,logs,qdrant}

# Run with custom settings
docker run -d \
  --name ai-inference \
  -p 3000:3000 \
  -e MAX_CONCURRENT_REQUESTS=20 \
  -e BATCH_MAX_SIZE=8 \
  -v ./data/models:/app/models \
  -v ./data/logs:/app/logs \
  --restart unless-stopped \
  ai-inference-server
```

### Docker Compose Profiles

| Profile | Services | Use Case |
|---------|----------|----------|
| Default | AI Server + Qdrant | Development |
| `production` | + Nginx reverse proxy | Production with load balancing |
| `monitoring` | + Prometheus + Grafana | Observability |
| `cache` | + Redis | Session management |

## ☸️ Option 2: Kubernetes Deployment (Production Ready)

### Prerequisites
```bash
# Verify cluster access
kubectl cluster-info

# Create namespace
kubectl apply -f k8s/ai-inference-deployment.yaml
```

### Deployment Steps

1. **Deploy Qdrant Database**
   ```bash
   kubectl apply -f k8s/qdrant/
   ```

2. **Wait for Qdrant to be Ready**
   ```bash
   kubectl wait --for=condition=available --timeout=300s deployment/qdrant -n qdrant-system
   ```

3. **Build and Push Container Image**
   ```bash
   # Build
   docker build -t your-registry/ai-inference-server:latest .
   
   # Push to your container registry
   docker push your-registry/ai-inference-server:latest
   
   # Update deployment with your image
   sed -i 's|ai-inference-server:latest|your-registry/ai-inference-server:latest|' k8s/ai-inference-deployment.yaml
   ```

4. **Deploy AI Inference Server**
   ```bash
   kubectl apply -f k8s/ai-inference-deployment.yaml
   ```

5. **Verify Deployment**
   ```bash
   # Check pods
   kubectl get pods -n ai-inference-system
   
   # Check service
   kubectl get svc -n ai-inference-system
   
   # Test health endpoint
   kubectl port-forward svc/ai-inference 3000:80 -n ai-inference-system
   curl http://localhost:3000/health
   ```

### Kubernetes Features

- ✅ **High Availability**: 3 replicas with anti-affinity
- ✅ **Auto-scaling**: HPA based on CPU/memory
- ✅ **Rolling Updates**: Zero-downtime deployments
- ✅ **Resource Management**: Requests and limits
- ✅ **Health Checks**: Liveness, readiness, startup probes
- ✅ **Security**: Non-root user, read-only filesystem
- ✅ **Monitoring**: Prometheus metrics integration

### GPU Support in Kubernetes

1. **Install NVIDIA Device Plugin**
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
   ```

2. **Enable GPU in Deployment**
   ```yaml
   resources:
     requests:
       nvidia.com/gpu: "1"
     limits:
       nvidia.com/gpu: "1"
   ```

## ☁️ Option 3: Cloud Platform Deployment

### AWS EKS Deployment

1. **Create EKS Cluster**
   ```bash
   eksctl create cluster --name ai-inference --version 1.24 --nodegroup-name standard-workers --node-type m5.xlarge --nodes 3
   ```

2. **Install Load Balancer Controller**
   ```bash
   kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"
   helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system
   ```

3. **Deploy with EBS Storage**
   ```bash
   # Update storage class in k8s manifests
   sed -i 's/fast-ssd/gp3/' k8s/ai-inference-deployment.yaml
   kubectl apply -f k8s/
   ```

### Google GKE Deployment

1. **Create GKE Cluster**
   ```bash
   gcloud container clusters create ai-inference \
     --machine-type n1-standard-4 \
     --num-nodes 3 \
     --enable-autoscaling \
     --min-nodes 1 \
     --max-nodes 10
   ```

2. **Deploy Application**
   ```bash
   kubectl apply -f k8s/
   ```

### Azure AKS Deployment

1. **Create AKS Cluster**
   ```bash
   az aks create \
     --resource-group ai-inference \
     --name ai-inference-cluster \
     --node-count 3 \
     --enable-addons monitoring \
     --generate-ssh-keys
   ```

2. **Configure kubectl**
   ```bash
   az aks get-credentials --resource-group ai-inference --name ai-inference-cluster
   ```

## 🔧 Configuration Management

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `3000` | Server port |
| `MODEL_CACHE_DIR` | `/app/models` | Model storage location |
| `MAX_CONCURRENT_REQUESTS` | `10` | Request concurrency limit |
| `BATCH_MAX_SIZE` | `4` | Batch processing size |
| `QDRANT_URL` | `http://localhost:6334` | Qdrant connection URL |
| `RAYON_NUM_THREADS` | `4` | CPU thread pool size |

### Secret Management

#### Docker Secrets
```bash
echo "your-secret" | docker secret create api-key -
```

#### Kubernetes Secrets
```bash
kubectl create secret generic ai-inference-secrets \
  --from-literal=api-key="your-secret" \
  -n ai-inference-system
```

## 📊 Monitoring and Observability

### Health Checks
- **Endpoint**: `GET /health`
- **Response**: `{"status": "healthy", "timestamp": "..."}`

### Metrics (with Prometheus)
- **Endpoint**: `GET /metrics`
- **Metrics**: Request latency, throughput, error rates, model performance

### Logging
- **Format**: Structured JSON
- **Levels**: ERROR, WARN, INFO, DEBUG
- **Storage**: Configurable (stdout, files, centralized)

## 🔐 Security Considerations

### Production Security Checklist
- [ ] Use non-root container user
- [ ] Enable read-only root filesystem
- [ ] Set resource limits
- [ ] Use secrets for sensitive data
- [ ] Enable TLS/HTTPS
- [ ] Configure network policies
- [ ] Regular security scans
- [ ] Access logging enabled

### Network Security
```yaml
# Example NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-inference-netpol
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: ai-inference
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 3000
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Model Loading Failures
```bash
# Check logs
docker logs ai-inference-server
kubectl logs -l app.kubernetes.io/name=ai-inference -n ai-inference-system

# Verify model cache
docker exec ai-inference-server ls -la /app/models
```

#### 2. Qdrant Connection Issues
```bash
# Test Qdrant connectivity
curl http://localhost:6333/health
kubectl port-forward svc/qdrant 6333:6333 -n qdrant-system
```

#### 3. Resource Issues
```bash
# Check resource usage
docker stats
kubectl top pods -n ai-inference-system
```

#### 4. GPU Not Detected
```bash
# Verify GPU availability
nvidia-smi
kubectl describe nodes | grep gpu
```

### Debug Commands

```bash
# Docker debugging
docker exec -it ai-inference-server /bin/sh

# Kubernetes debugging
kubectl exec -it deployment/ai-inference -n ai-inference-system -- /bin/sh
kubectl describe pod <pod-name> -n ai-inference-system
kubectl logs <pod-name> -n ai-inference-system --previous
```

## 📈 Scaling and Performance

### Horizontal Scaling
```bash
# Docker Compose
docker-compose up -d --scale ai-inference=3

# Kubernetes
kubectl scale deployment ai-inference --replicas=5 -n ai-inference-system
```

### Performance Tuning

#### CPU Optimization
```bash
# Set optimal thread count
export RAYON_NUM_THREADS=$(nproc)
export BATCH_MAX_SIZE=8
```

#### Memory Optimization
```bash
# Configure batch sizes based on available memory
# 16GB RAM: BATCH_MAX_SIZE=8
# 32GB RAM: BATCH_MAX_SIZE=16
```

#### GPU Optimization
```bash
# Enable GPU features
export CUDA_VISIBLE_DEVICES=0
# Build with: --build-arg FEATURES=cuda
```

## 📱 Testing Your Deployment

### Basic Functionality Test
```bash
# Health check
curl http://localhost:3000/health

# Simple generation
curl -X POST http://localhost:3000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'

# Model selection
curl -X POST http://localhost:3000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain AI", "model": "tinyllama", "temperature": 0.7}'
```

### Load Testing
```bash
# Install Apache Bench
apt-get install apache2-utils

# Run load test
ab -n 1000 -c 10 -H "Content-Type: application/json" \
  -p test-payload.json \
  http://localhost:3000/api/v1/generate
```

### Vector Operations Test
```bash
# Test vector storage
curl -X POST http://localhost:3000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "metadata": {"text": "test"}}'

# Test semantic search
curl -X POST http://localhost:3000/api/v1/vectors/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "search text", "limit": 5}'
```

## 🔄 Updates and Maintenance

### Rolling Updates
```bash
# Docker Compose
docker-compose pull
docker-compose up -d

# Kubernetes
kubectl set image deployment/ai-inference ai-inference=your-registry/ai-inference-server:v2.0.0 -n ai-inference-system
kubectl rollout status deployment/ai-inference -n ai-inference-system
```

### Backup and Recovery
```bash
# Backup Qdrant data
kubectl exec deployment/qdrant -n qdrant-system -- curl -X POST http://localhost:6333/snapshots

# Backup models
kubectl cp ai-inference-system/ai-inference-xxx:/app/models ./models-backup
```

## 📞 Support and Resources

### Documentation
- [API Documentation](./API_DOCUMENTATION.md)
- [Development Roadmap](./DEVELOPMENT_ROADMAP.md)
- [Architecture Guide](./CLAUDE.md)

### Monitoring Dashboards
- **Grafana**: http://localhost:3001 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Qdrant UI**: http://localhost:6333/dashboard

### Community and Support
- Report issues on GitHub
- Check logs for detailed error messages
- Monitor resource usage during peak loads

---

## ✅ Deployment Checklist

### Pre-deployment
- [ ] Hardware requirements met
- [ ] Docker/Kubernetes installed
- [ ] Container registry access configured
- [ ] Secrets and configurations prepared

### Deployment
- [ ] Services deployed successfully
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Load testing completed

### Post-deployment
- [ ] Performance monitoring active
- [ ] Backup procedures tested
- [ ] Alert thresholds configured
- [ ] Documentation updated

**Your AI Inference Server is now ready for production! 🎉**