---
title: AI Inference Server
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 3000
pinned: false
license: mit
---

# 🚀 AI Inference Server

A high-performance AI inference server built with Rust and Candle framework, optimized for concurrent model inference with enterprise-grade features.

## ✨ Features

- **Multi-Model Support**: TinyLlama, Llama-2, Generic Llama, Phi-3, Gemma
- **High Performance**: 10-14 tokens/second on optimized hardware
- **Concurrent Inference**: Lock-free architecture for multiple simultaneous requests
- **Vector Database**: Qdrant integration for semantic search and RAG
- **Fast Sampling**: Optimized top-k sampling with temperature control
- **Memory Efficient**: Per-request cache management
- **Production Ready**: Health checks, monitoring, and failover capabilities

## 🚀 Quick Start

### API Endpoints

#### Text Generation
```bash
curl -X POST https://your-space.hf.space/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "model": "tinyllama"}'
```

#### Model Selection
```bash
# TinyLlama (fast, 1.1B params)
curl -X POST https://your-space.hf.space/api/v1/generate \
  -d '{"prompt": "Explain AI", "model": "tinyllama"}'

# Generic Llama (flexible)
curl -X POST https://your-space.hf.space/api/v1/generate \
  -d '{"prompt": "What is Rust?", "model": "llama-generic"}'
```

#### File Upload Generation
```bash
curl -X POST https://your-space.hf.space/api/v1/generate/upload \
  -F "file=@document.txt" \
  -F "prompt=Summarize this document" \
  -F "model=tinyllama"
```

## API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "ai-inference-server", 
  "version": "0.1.0",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Text Generation
```http
POST /v1/generate
Content-Type: application/json

{
  "prompt": "Hello, how are you?",
  "max_tokens": 100,
  "temperature": 0.7,
  "model": "microsoft/DialoGPT-medium"
}
```

Response:
```json
{
  "id": "uuid-here",
  "text": "Generated response text",
  "model": "microsoft/DialoGPT-medium",
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 100,
    "total_tokens": 104
  }
}
```

## Development

### Project Structure

```
src/
├── lib.rs              # Module declarations
├── config.rs           # Configuration management
├── error.rs            # Error types and handling
├── api/
│   ├── mod.rs          # API router setup
│   ├── health.rs       # Health check endpoints
│   └── generate.rs     # Text generation endpoints
└── models/
    └── mod.rs          # Model loading and management
```

### Running Tests

```bash
cargo test
```

### Linting

```bash
cargo clippy
cargo fmt
```

## Dependencies

- **axum**: Modern web framework
- **tokio**: Async runtime
- **candle-core**: ML tensor operations
- **candle-nn**: Neural network layers
- **candle-transformers**: Transformer models
- **tokenizers**: Text tokenization
- **serde**: Serialization
- **tracing**: Structured logging
- **anyhow**: Error handling
- **uuid**: Request ID generation
- **hf-hub**: Model downloading

## License

MIT License - see LICENSE file for details.