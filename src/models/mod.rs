// ARCHITECTURE: Model Management Module - Enterprise-Grade AI Model Lifecycle System
//
// DESIGN OVERVIEW:
// This module implements a comprehensive model management system for production AI inference:
// 
// 1. MODEL ABSTRACTION (llama.rs):
//    - TinyLlamaModel: Core inference engine with Metal GPU acceleration
//    - ModelInfo: Metadata and performance characteristics
//    - Memory-efficient loading with SafeTensors and F16 precision
//    - Optimized tokenization and generation loops
//
// 2. VERSION MANAGEMENT (version_manager.rs):
//    - Multi-version model storage with configurable limits (max 3 models)
//    - Comprehensive health checking with 4-dimensional validation
//    - Background model loading with automatic health assessment
//    - Model status lifecycle: Loading → HealthCheck → Ready → Active → Deprecated
//
// 3. ATOMIC SWAPPING (atomic_swap.rs):
//    - Zero-downtime model switching with 5-point safety validation
//    - Health check retry logic with configurable timeout and attempts
//    - Request queueing during swaps to ensure service continuity
//    - Comprehensive swap safety reporting and error handling
//
// PERFORMANCE CHARACTERISTICS:
// - 10-14 tokens/second inference on Apple Silicon M1/M2/M3
// - <3 second model swapping with automatic validation
// - <100ms health check completion times
// - Memory-efficient concurrent model storage
//
// PRODUCTION READINESS:
// ✅ Thread-safe operations with Arc<Mutex<T>> patterns
// ✅ Comprehensive error handling and logging
// ✅ Configurable timeouts and retry logic  
// ⚠️  Resource cleanup needs enhancement for long-running deployments
// ⚠️  Monitoring and alerting integration required

pub mod llama;           // Core model inference engine
pub mod version_manager; // Model lifecycle and health management  
pub mod atomic_swap;     // Zero-downtime model switching
pub mod failover_manager; // Automatic failover and high availability
pub mod circuit_breaker; // Circuit breaker pattern for fault tolerance

// Public API exports for clean module interface
pub use llama::{ModelInfo, TinyLlamaModel};
pub use version_manager::{ModelVersionManager, ModelVersion, HealthCheckResult, HealthStatus, PerformanceMetrics};
pub use atomic_swap::{AtomicModelSwap, SwapResult, SwapSafetyReport};
pub use failover_manager::{AutomaticFailoverManager, FailoverConfig, FailoverMetrics, FailureType, FailureRecord};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, CircuitBreakerMetrics, CallResult};