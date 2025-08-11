# AI Inference Server

A high-performance AI inference server built with Rust, using Candle for ML operations and Axum for HTTP serving.

## Features

- **Fast HTTP API** using Axum web framework
- **Async processing** with Tokio runtime
- **ML inference** powered by Candle
- **Text generation** with transformer models
- **Token management** with HuggingFace tokenizers
- **Model caching** for improved performance
- **Health monitoring** and observability
- **Configurable** via environment variables

## Prerequisites

- Rust 1.70+ with Cargo
- CUDA (optional, for GPU acceleration)

## Setup Instructions

### 1. Clone and Build

```bash
git clone <repository-url>
cd ai-inference-server
cargo build --release
```

### 2. Configuration

Set environment variables or use defaults:

```bash
# Server configuration
export HOST=0.0.0.0
export PORT=8080
export LOG_LEVEL=info

# Model configuration
export MODEL_CACHE_DIR=./models
export MAX_CONCURRENT_REQUESTS=10
export DEFAULT_MODEL=microsoft/DialoGPT-medium
```

### 3. Run the Server

```bash
cargo run --release
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