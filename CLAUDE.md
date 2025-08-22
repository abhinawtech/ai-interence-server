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

# Run specific test files (integration tests in tests/ directory)
cargo test --test test_day9_1_index_optimization
cargo test --test test_day9_2_background_reindexing 
cargo test --test test_day9_3_index_monitoring
cargo test --test test_day9_3_simple

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

### Frontend Development
```bash
# Navigate to frontend directory and install dependencies
cd frontend && npm install

# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Lint frontend code
npm run lint

# Preview production build
npm run preview
```

## Architecture Overview

### Core System Design
This is an enterprise-grade AI inference server with advanced vector database capabilities and a modern React frontend. The system follows a modular architecture with clear separation of concerns:

- **AI Inference Engine**: Candle-powered ML inference with hot model swapping
- **Vector Database**: Qdrant integration for semantic search and embeddings
- **Index Management**: Production-ready indexing optimization and monitoring
- **Async Architecture**: Tokio-based async runtime with batch processing
- **Frontend Interface**: React + TypeScript + Vite frontend with Tailwind CSS and Radix UI components

### Key Modules

#### `src/api/` - RESTful API Layer
- `generate.rs`: Core text generation with memory integration and dynamic model selection
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

#### `frontend/` - React Web Interface
- **Framework Stack**: React 19 + TypeScript + Vite for fast development and builds
- **Styling**: Tailwind CSS for utility-first styling with PostCSS processing
- **UI Components**: Radix UI primitives for accessible, unstyled components
- **Icons**: Lucide React for consistent iconography
- **Build System**: Vite with hot module replacement and optimized production builds

### Critical Architecture Patterns

#### Model Management Pattern
- Models are loaded via `ModelVersionManager` with health checking
- `AtomicModelSwap` enables zero-downtime model updates
- Dynamic model selection through API requests with intelligent switching logic
- Failover capabilities handle model loading failures gracefully
- Model aliases allow user-friendly names (e.g., "tinyllama" → "tinyllama-1.1b-chat")

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

#### Unified State Architecture
The `GenerateState` type encapsulates all core services:
```rust
pub type GenerateState = (
    Arc<BatchProcessor>,                    // Text generation processing
    Arc<VectorBackend>,                     // Vector storage for RAG
    Arc<RwLock<EmbeddingService>>,         // Embedding generation
    Arc<SearchSessionManager>,             // Session management for memory
    Arc<Mutex<DocumentIngestionPipeline>>, // Document processing pipeline
    Arc<IntelligentChunker>,               // Document chunking
    Arc<ModelVersionManager>,              // Model version management
);
```
This tuple-based state enables dependency injection across all API endpoints while maintaining type safety and enabling hot-swappable components.

## Development Context

### Performance Characteristics
- **Inference Speed**: 10-14 tokens/second on Apple Silicon (Metal GPU)
- **Latency**: Sub-100ms for single requests via fast-path optimization
- **Throughput**: 2-4x improvement with intelligent batching
- **Model Swapping**: <3 second hot swaps with automatic validation

### Advanced Features

#### Model Selection
- Dynamic model switching via API parameter: `"model": "tinyllama"` or `"model": "gemma"`
- Support for model aliases (e.g., "tinyllama", "llama-generic", "gemma")
- Intelligent model loading with timeout handling and health validation
- Backwards compatibility - omitting model parameter uses currently active model

#### Vector Database Features
The system implements a comprehensive vector database solution:

- **Foundation**: Basic vector operations, semantic search, memory integration
- **Index Management**: Index optimization (3 performance profiles), background reindexing system, comprehensive monitoring with alerting
- **Document Processing**: Complete document ingestion pipeline with intelligent chunking, incremental updates, and deduplication

### Configuration

#### Backend Configuration
Primary configuration via environment variables:
- `HOST`, `PORT`: Server binding (defaults to `0.0.0.0:3000`)
- `MODEL_CACHE_DIR`: Model storage location
- `MAX_CONCURRENT_REQUESTS`: Request throttling
- `RAYON_NUM_THREADS`: Compute thread pool size (defaults to 4)
- `BATCH_MAX_SIZE`: Maximum batch size for request processing (defaults to 4)
- `BATCH_MAX_WAIT_MS`: Maximum wait time for batching in milliseconds (defaults to 100)
- `BATCH_MAX_QUEUE_SIZE`: Maximum queue size for batch processing (defaults to 50)
- `TOKENIZERS_PARALLELISM`: Set to "false" to avoid conflicts with Tokio async runtime

The server runs on port 3000 by default. Test endpoints with:
```bash
curl http://localhost:3000/health
curl -X POST http://localhost:3000/api/v1/generate -H "Content-Type: application/json" -d '{"prompt": "Hello"}'

# Test with model selection
curl -X POST http://localhost:3000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "model": "tinyllama"}'

# Test file upload generation
curl -X POST http://localhost:3000/api/v1/generate/upload \
  -F "file=@test.txt" \
  -F "prompt=Summarize this document" \
  -F "model=tinyllama"
```

#### Frontend Configuration
- Vite development server runs on port 5173 by default
- Production builds are output to `frontend/dist/` directory
- TypeScript configuration via `tsconfig.json` and `tsconfig.app.json`
- ESLint configuration in `eslint.config.js` with React and TypeScript rules

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
  - `test_day9_3_simple.rs`: Simplified integration tests
- Model selection can be tested with curl commands like:
  ```bash
  curl -X POST http://localhost:3000/api/v1/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "model": "tinyllama"}'
  ```
- Performance benchmarking via simulation methods
- Test files in project root (e.g., `simple_test.txt`, `test_upload.txt`) for upload testing

### API Endpoints
The server provides comprehensive REST APIs organized by functionality:

#### Core AI Services
- Text generation: `/api/v1/generate` (supports model selection via `"model"` parameter)
- File upload generation: `/api/v1/generate/upload` (supports model selection via form field)
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

#### Document Processing
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

## Important Files and Directories

### Configuration Files
- `Cargo.toml`: Rust project configuration with optimized build settings and GPU acceleration features
- `docker-compose.qdrant.yml`: Local Qdrant vector database setup with health checks and performance tuning
- `config/qdrant/production.yaml`: Qdrant production configuration
- `frontend/package.json`: Frontend dependencies (React 19, TypeScript, Vite, Tailwind CSS, Radix UI)

### Development Files
- `API_DOCUMENTATION.md`: Complete API documentation with request/response examples
- `DEVELOPMENT_ROADMAP.md`: Project development timeline and feature roadmap
- `examples/`: API testing guides and documentation
- `test_documents/`: Sample documents for testing document processing features

### Data Directories
- `data/qdrant/`: Persistent storage for Qdrant vector database
- `qdrant_storage/`: Alternative Qdrant storage location
- `target/`: Rust build artifacts (gitignored)