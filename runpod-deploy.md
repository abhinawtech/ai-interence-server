# RunPod GPU Deployment Guide

## Why RunPod?
- ✅ $0.2/hour for RTX 3070 (cheapest GPU option)
- ✅ Docker support
- ✅ Pay only when running
- ✅ Auto-scaling

## Quick Deploy Steps

### 1. Sign up at runpod.io
- Get $10 free credit

### 2. Create GPU Dockerfile
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Rust and dependencies
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . .

# Build with CUDA support
RUN cargo build --release --features cuda

EXPOSE 3000
CMD ["./target/release/ai-interence-server"]
```

### 3. Deploy
1. Go to RunPod Console
2. Click "Deploy"
3. Select GPU (RTX 3070 recommended)
4. Upload your Docker image or connect GitHub
5. Set port: 3000
6. Deploy!

### 4. Configure
Set these environment variables:
```
CUDA_VISIBLE_DEVICES=0
RAYON_NUM_THREADS=4
BATCH_MAX_SIZE=8
```

## Cost Estimation
- RTX 3070: $0.2/hour = $144/month (24/7)
- RTX 4090: $0.4/hour = $288/month (24/7)
- A100: $1.2/hour = $864/month (24/7)

## Auto-scaling
Configure RunPod to:
- Scale to 0 when no requests
- Auto-start when requests come in
- Save ~70% on costs