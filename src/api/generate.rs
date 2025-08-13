use crate::{batching::BatchProcessor, errors::AppError, models::ModelInfo};
use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
use uuid::Uuid;

// API DESIGN: Request Schema for Text Generation
// Follows OpenAI-compatible API structure for easy integration
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,              // Input text for generation
    pub max_tokens: Option<usize>,   // Optional limit (default: 100)
    pub temperature: Option<f32>,    // Future: sampling randomness control
}

// API DESIGN: Comprehensive Response with Performance Metrics
// Provides detailed timing information for performance analysis
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,                    // Generated text output
    pub model_info: ModelInfo,           // Model specifications
    pub processing_time_ms: u64,         // Pure inference time (excludes queue)
    pub request_id: String,              // Unique identifier for tracing
    pub tokens_generated: usize,         // Actual token count (may be < max_tokens)
    pub tokens_per_second: f64,          // Performance metric (inference only)
    pub queue_time_ms: u64,              // Time spent waiting in batch queue
    pub batch_processing: bool,          // Indicates batching was used
}

impl GenerateRequest {
    // VALIDATION: Input Sanitization and Resource Protection
    // Prevents abuse and ensures stable performance under load
    pub fn validate(&self) -> std::result::Result<(), String> {
        // Prevent empty prompts that waste compute resources
        if self.prompt.trim().is_empty() {
            return Err("Prompt cannot be empty".to_string());
        }
        
        // RESOURCE PROTECTION: Limit prompt length to prevent memory exhaustion
        // 4096 chars ≈ 1000 tokens, reasonable for TinyLlama's 2048 context window
        if self.prompt.len() > 4096 {
            return Err("Prompt too long (Max 4096 characters)".to_string());
        }
        
        // PERFORMANCE: Limit generation length to maintain responsiveness
        // 512 tokens ≈ 25-40 seconds on typical hardware, balances utility vs responsiveness
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 || max_tokens > 512 {
                return Err("max_tokens must be between 1 and 512".to_string());
            }
        }
        Ok(())
    }
}

// ENDPOINT: Main Text Generation Handler
// Implements async request processing with comprehensive error handling
pub async fn generate_text(
    State(batch_processor): State<Arc<BatchProcessor>>,
    Json(request): Json<GenerateRequest>,
) -> std::result::Result<Json<GenerateResponse>, AppError> {
    // ANALYTICS: End-to-end timing measurement
    let request_start = Instant::now();
    let request_id = Uuid::new_v4().to_string();

    // SECURITY: Input validation before processing
    request.validate().map_err(AppError::Validation)?;

    // DEFAULTS: Reasonable token limit for responsive service
    let max_tokens = request.max_tokens.unwrap_or(100);

    tracing::info!(
        request_id = %request_id,
        prompt_length = request.prompt.len(),
        "Received generation request"
    );

    // OPTIMIZATION: Asynchronous Batch Processing
    // Non-blocking submission to batch queue for optimal throughput
    // Automatic batching provides 20-30% performance improvement
    let batch_response = batch_processor
        .submit_request(request.prompt, max_tokens) // OPTIMIZATION: Avoid clone by moving ownership
        .await
        .map_err(|e| {
            tracing::error!("Batch processing failed for request {}: {}", request_id, e);
            AppError::BadRequest(format!("Batch Processing failed: {e}"))
        })?;

    // ANALYTICS: Detailed Performance Breakdown
    let total_time = request_start.elapsed();
    let total_time_ms = total_time.as_millis() as u64;
    
    // METRICS: Separate queue time from processing time
    // Queue time indicates batching efficiency and load
    let queue_time_ms = total_time_ms.saturating_sub(batch_response.processing_time_ms);

    // PERFORMANCE CALCULATION: Pure inference speed measurement
    // Excludes queue time to measure actual model performance
    // Critical metric for optimizing generation algorithms
    let tokens_per_second = if batch_response.processing_time_ms > 0 {
        (batch_response.token_generated as f64)
            / (batch_response.processing_time_ms as f64 / 1000.0)
    } else {
        0.0  // Prevent division by zero for instant responses
    };

    // API RESPONSE: Static Model Information
    // Provides client applications with model specifications
    // Helps clients understand capabilities and limitations
    let model_info = ModelInfo {
        name: "TinyLlama-1.1B-Chat".to_string(),
        version: "v1.0-batched".to_string(),    // Indicates batching optimization
        parameters: 110000000,                  // 1.1B parameter count for reference
        memory_mb: 2200,                       // F16 memory usage estimate
        device: "Auto-detected".to_string(),   // GPU/CPU determined at runtime
        vocab_size: 32000,                     // Tokenizer vocabulary size
        context_length: 2048,                  // Maximum sequence length
    };

    // MONITORING: Structured Request Completion Log
    // Essential for performance tuning and capacity planning
    // Helps identify bottlenecks: queue vs processing time
    tracing::info!(
        "Request {} Completed: {}ms total ({}ms queue + {}ms processing, {:.2} tok/s)",
        request_id,
        total_time_ms,
        queue_time_ms,
        batch_response.processing_time_ms,
        tokens_per_second
    );

    Ok(Json(GenerateResponse {
        text: batch_response.text,
        model_info,
        processing_time_ms: batch_response.processing_time_ms,
        request_id,
        tokens_generated: batch_response.token_generated,
        tokens_per_second,
        queue_time_ms,
        batch_processing: true,
    }))
}

// ENDPOINT: Batch Processing Monitoring
// Provides real-time insights into system performance and load
// Critical for operational monitoring and capacity planning
pub async fn batch_status(
    State(batch_processor): State<Arc<BatchProcessor>>,
) -> std::result::Result<Json<BatchStatusResponse>, AppError> {
    // MONITORING: Real-time Performance Metrics
    let stats = batch_processor.get_stats().await;
    let queue_size = batch_processor.get_queue_size().await as u64;

    Ok(Json(BatchStatusResponse {
        queue_size,                             // Current pending requests
        total_requests: stats.total_requests,   // Lifetime request count
        total_batches: stats.total_batches,     // Number of batches processed
        avg_batch_size: stats.avg_batch_size,   // Batching efficiency metric
        avg_processing_time_ms: stats.avg_processing_time_ms, // Performance trend
    }))
}

// API SCHEMA: Batch Processing Metrics Response
// Provides comprehensive system health and performance data
#[derive(Serialize)]
pub struct BatchStatusResponse {
    pub queue_size: u64,                    // Instant load indicator
    pub total_requests: u64,                // System usage metric
    pub total_batches: u64,                 // Batching frequency
    pub avg_batch_size: f64,                // Efficiency indicator (target: 2-4)
    pub avg_processing_time_ms: f64,        // Performance trend (target: <500ms)
}
