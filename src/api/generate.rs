use crate::{batching::BatchProcessor, errors::AppError, models::ModelInfo};
use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub model_info: ModelInfo,
    pub processing_time_ms: u64,
    pub request_id: String,
    pub tokens_generated: usize,
    pub tokens_per_second: f64,
    pub queue_time_ms: u64,
    pub batch_processing: bool,
}

impl GenerateRequest {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.prompt.trim().is_empty() {
            return Err("Prompt cannot be empty".to_string());
        }
        if self.prompt.len() > 4096 {
            return Err("Promt too long (Max 4096 char)".to_string());
        }
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 || max_tokens > 512 {
                return Err("max_tokens must be between 1 and 512".to_string());
            }
        }
        Ok(())
    }
}

pub async fn generate_text(
    State(batch_processor): State<Arc<BatchProcessor>>,
    Json(request): Json<GenerateRequest>,
) -> std::result::Result<Json<GenerateResponse>, AppError> {
    let request_start = Instant::now();
    let request_id = Uuid::new_v4().to_string();

    request.validate().map_err(AppError::Validation)?;

    let max_tokens = request.max_tokens.unwrap_or(100);

    tracing::info!(
        request_id = %request_id,
        prompt_length = request.prompt.len(),
        "Received generation request"
    );

    // Submit to batch processor
    let batch_response = batch_processor
        .submit_request(request.prompt.clone(), max_tokens)
        .await
        .map_err(|e| {
            tracing::error!("Batch processing failed for request {}: {}", request_id, e);
            AppError::BadRequest(format!("Batch Processing failed: {e}"))
        })?;

    let total_time = request_start.elapsed();
    let total_time_ms = total_time.as_millis() as u64;
    let queue_time_ms = total_time_ms.saturating_sub(batch_response.processing_time_ms);

    // Calculate tokens per second based on processing time (not incuding queue time)
    let tokens_per_second = if batch_response.processing_time_ms > 0 {
        (batch_response.token_generated as f64)
            / (batch_response.processing_time_ms as f64 / 1000.0)
    } else {
        0.0
    };

    // Create model info simplified for batching
    let model_info = ModelInfo {
        name: "TinyLlama-1.1B-Chat".to_string(),
        version: "v1.0-batched".to_string(),
        parameters: 110000000,
        memory_mb: 2200,
        device: "Auto-detected".to_string(),
        vocab_size: 32000,
        context_length: 2048,
    };

    tracing::info!(
        "Request {} Completed: {}ms total ({}ms queue + {}ms processing, {:.2} tok/s",
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

pub async fn batch_status(
    State(batch_processor): State<Arc<BatchProcessor>>,
) -> std::result::Result<Json<BatchStatusResponse>, AppError> {
    let stats = batch_processor.get_stats().await;
    let queue_size = batch_processor.get_queue_size().await as u64;

    Ok(Json(BatchStatusResponse {
        queue_size,
        total_requests: stats.total_requests,
        total_batches: stats.total_batches,
        avg_batch_size: stats.avg_batch_size,
        avg_processing_time_ms: stats.avg_processing_time_ms,
    }))
}

#[derive(Serialize)]
pub struct BatchStatusResponse {
    pub queue_size: u64,
    pub total_requests: u64,
    pub total_batches: u64,
    pub avg_batch_size: f64,
    pub avg_processing_time_ms: f64,
}
