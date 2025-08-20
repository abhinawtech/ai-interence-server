# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Build and Development
```bash
# Development build with optimizations
cargo build

# Release build with maximum optimization
cargo build --release

# Run the server
cargo run --release

# Run with CUDA GPU acceleration (if available)
cargo run --release --features cuda
```

### Testing
```bash
# Run all tests
cargo test

# Run specific test file
cargo test --test test_day9_1_index_optimization

# Run tests with output
cargo test -- --nocapture

# Run tests quietly (less output)
cargo test --quiet
```

### Code Quality
```bash
# Lint code
cargo clippy

# Format code
cargo fmt

# Check compilation without building
cargo check
```

### Development with Vector Database
```bash
# Start Qdrant vector database (required for vector operations)
docker-compose -f docker-compose.qdrant.yml up -d

# Stop Qdrant
docker-compose -f docker-compose.qdrant.yml down
```

## Architecture Overview

### Core System Design
This is an enterprise-grade AI inference server with advanced vector database capabilities. The system follows a modular architecture with clear separation of concerns:

- **AI Inference Engine**: Candle-powered ML inference with hot model swapping
- **Vector Database**: Qdrant integration for semantic search and embeddings
- **Index Management**: Production-ready indexing optimization and monitoring
- **Async Architecture**: Tokio-based async runtime with batch processing

### Key Modules

#### `src/api/` - RESTful API Layer
- `generate.rs`: Core text generation with memory integration
- `vectors.rs` & `vectors_enhanced.rs`: Vector database operations
- `embedding.rs`: Text-to-vector conversion services  
- `search.rs`: Session-aware semantic search with context
- `index_management.rs`: Index optimization, background reindexing, and monitoring APIs
- `models.rs`: Model lifecycle management with zero-downtime swapping
- `document_processing.rs`: Document ingestion, chunking, and deduplication APIs

#### `src/models/` - ML Model Management
- `version_manager.rs`: Hot model swapping with health validation
- `atomic_swap.rs`: Zero-downtime model updates
- `registry.rs`: Model discovery and configuration
- Individual model implementations (`llama.rs`, `phi3.rs`, etc.)

#### `src/vector/` - Vector Database & Search
- **Storage Layer**: `storage.rs`, `qdrant_operations.rs` - Vector CRUD operations
- **Embedding Services**: `embedding_service.rs` - Text-to-vector conversion
- **Index Optimization**: `index_optimizer.rs` - Performance profile management (HighAccuracy/Balanced/FastQuery)  
- **Background Jobs**: `reindex_manager.rs` - Zero-downtime reindexing with job queues
- **Monitoring**: `index_monitor.rs` - Real-time performance metrics and alerting
- **Document Processing**: `document_ingestion.rs`, `chunking_strategies.rs`, `incremental_updates.rs` - Document pipeline and intelligent chunking

#### `src/batching/` - Performance Optimization
Intelligent request batching for 2-4x throughput improvement on batch workloads.

#### `src/security/` - Authentication & Rate Limiting
- `rate_limiter.rs`: Request throttling and backpressure
- `auth.rs`: Authentication middleware (extensible)

### Critical Architecture Patterns

#### Model Management Pattern
- Models are loaded via `ModelVersionManager` with health checking
- `AtomicModelSwap` enables zero-downtime model updates
- Failover capabilities handle model loading failures gracefully

#### Vector Database Integration
- Factory pattern (`VectorStorageFactory`) abstracts storage backends
- Embedding pipeline: Text → Tokenizer → Model → Vector → Storage
- Three-tier optimization: Collection optimization → Background reindexing → Performance monitoring

#### Session Management
- `SearchSessionManager` maintains conversation context
- Memory integration enables contextual AI responses
- Session-aware search with personalized recommendations

#### Background Job System
- `ReindexManager` handles long-running index optimization tasks
- Priority-based job queuing with pause/resume/cancel capabilities
- Resource usage monitoring and automatic queue management

#### Document Processing Pipeline
- `DocumentIngestionPipeline` handles multi-format document parsing (PDF, DOCX, TXT, MD)
- `IntelligentChunker` provides semantic, sentence, and token-based chunking strategies
- `IncrementalUpdateManager` supports incremental document updates and deduplication
- Automatic metadata extraction and version tracking

## Development Context

### Performance Characteristics
- **Inference Speed**: 10-14 tokens/second on Apple Silicon (Metal GPU)
- **Latency**: Sub-100ms for single requests via fast-path optimization
- **Throughput**: 2-4x improvement with intelligent batching
- **Model Swapping**: <3 second hot swaps with automatic validation

### Vector Database Features
The system implements a comprehensive vector database solution:

- **Day 6-8 Implementation**: Basic vector operations, semantic search, memory integration
- **Day 9 Implementation**: Index optimization (3 performance profiles), background reindexing system, comprehensive monitoring with alerting
- **Day 10 Implementation**: Complete document ingestion pipeline with intelligent chunking, incremental updates, and deduplication

### Configuration
Primary configuration via environment variables:
- `HOST`, `PORT`: Server binding
- `MODEL_CACHE_DIR`: Model storage location
- `MAX_CONCURRENT_REQUESTS`: Request throttling
- `RAYON_NUM_THREADS`: Compute thread pool size

### Error Handling
- `AppError` enum provides structured error responses
- Graceful degradation when vector database is unavailable
- Circuit breaker pattern for external service failures

### Testing Strategy
- Unit tests for individual components (`cargo test`)
- Integration tests for API endpoints (`tests/` directory)
- Specific test suites for each major feature:
  - `test_day9_1_index_optimization.rs`: Index optimization testing
  - `test_day9_2_background_reindexing.rs`: Background reindexing system
  - `test_day9_3_index_monitoring.rs`: Performance monitoring and alerting
- Document processing tests: `test_day10_simple.rs`, `upload_test_document.rs`
- Performance benchmarking via simulation methods

### API Endpoints
The server provides comprehensive REST APIs organized by functionality:

#### Core AI Services
- Text generation: `/api/v1/generate`
- Model management: `/api/v1/models/*` 
- Health checks: `/health`

#### Vector Database Operations
- Vector CRUD: `/api/v1/vectors/*`
- Semantic search: `/api/v1/search/*`
- Embedding services: `/api/v1/embeddings/*`

#### Index Management  
- Index optimization: `/api/v1/index/optimize`
- Background reindexing: `/api/v1/index/reindex`
- Performance monitoring: `/api/v1/index/monitor/*`

#### Document Processing (Day 10)
- Document ingestion: `/api/v1/documents/ingest`
- File upload: `/api/v1/documents/upload`
- Document retrieval: `/api/v1/documents/{id}`
- Batch processing: `/api/v1/documents/ingest/batch`
- Chunk existing document: `/api/v1/documents/{id}/chunk`
- Chunk arbitrary content: `/api/v1/documents/chunk`
- Incremental updates: `/api/v1/documents/update`
- Deduplication: `/api/v1/documents/deduplicate`

### Kubernetes Deployment
Ready-to-use Kubernetes manifests in `k8s/` directory for:
- Qdrant vector database deployment with persistent storage
- Namespace isolation and service configuration
- ConfigMap-based configuration management