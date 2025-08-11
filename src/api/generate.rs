use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::sync::Arc;
use crate::{models::TinyLlamaModel, error::Result};

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub id: String,
    pub text: String,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub async fn generate_text(
    State(model): State<Arc<tokio::sync::Mutex<TinyLlamaModel>>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>> {
    let request_id = Uuid::new_v4().to_string();
    
    tracing::info!(
        request_id = %request_id,
        prompt_length = request.prompt.len(),
        "Received generation request"
    );

    let model_name = request.model.as_deref().unwrap_or("tinyllama");
    let max_tokens = request.max_tokens.unwrap_or(50); // Smaller default for faster response

    let mut model_guard = model.lock().await;
    let generated_text = model_guard
        .generate(&request.prompt, max_tokens)
        .await?;

    let response = GenerateResponse {
        id: request_id,
        text: generated_text,
        model: model_name.to_string(),
        usage: Usage {
            prompt_tokens: request.prompt.split_whitespace().count(),
            completion_tokens: max_tokens,
            total_tokens: request.prompt.split_whitespace().count() + max_tokens,
        },
    };

    Ok(Json(response))
}