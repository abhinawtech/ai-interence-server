// ARCHITECTURE: API Module - RESTful Service Interface Layer
//
// DESIGN PHILOSOPHY:
// This module implements a clean, well-structured REST API with the following principles:
// 1. SEPARATION OF CONCERNS: Dedicated modules for different API functionalities
// 2. SCALABILITY: Stateless design with efficient request handling
// 3. RELIABILITY: Comprehensive error handling and validation
// 4. OBSERVABILITY: Structured logging and performance monitoring
//
// API STRUCTURE:
// 1. GENERATE (generate.rs): High-performance inference endpoints
//    - POST /api/v1/generate: Text generation with batching optimization
//    - WebSocket support for streaming responses (future)
//    - Request validation and rate limiting
//
// 2. HEALTH (health.rs): System monitoring and health checks  
//    - GET /health: Basic service health indicator
//    - Load balancer integration and uptime monitoring
//    - Service discovery support
//
// 3. MODELS (models.rs): Model lifecycle management
//    - CRUD operations for model versions
//    - Hot swapping with zero-downtime guarantees
//    - Health checking and performance monitoring
//    - System status and operational visibility
//
// PRODUCTION READINESS:
// ✅ RESTful design with proper HTTP status codes
// ✅ Comprehensive input validation and sanitization
// ✅ Structured error responses with detailed context
// ✅ Performance monitoring and request tracing
// ⚠️  Rate limiting and authentication not implemented
// ⚠️  API versioning strategy needs formalization

pub mod generate;         // Inference endpoints - core text generation functionality
pub mod health;          // Health monitoring - service status and uptime checks  
pub mod models;          // Model management - lifecycle, swapping, and administration
pub mod vectors;         // Vector database operations - simple implementation
pub mod vectors_enhanced; // Enhanced vector operations - semantic search and filtering
pub mod embedding;       // Text-to-vector embedding API - semantic processing
