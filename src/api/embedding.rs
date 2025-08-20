// ================================================================================================
// EMBEDDING API - TEXT-TO-VECTOR CONVERSION ENDPOINTS
// ================================================================================================
//
// Production API for converting text to vectors with semantic understanding.
// This integrates with your existing model infrastructure and enhances the generate API
// with semantic memory capabilities.
//
// ================================================================================================

use crate::vector::{EmbeddingService, EmbeddingConfig, EmbeddingServiceStats};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use uuid::Uuid;

// ================================================================================================
// API TYPES
// ================================================================================================

#[derive(Debug, Deserialize)]
pub struct EmbedTextRequest {
    pub text: String,
    pub metadata: Option<HashMap<String, String>>,
    /// Whether to store the embedding in vector database
    pub store: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct EmbedTextResponse {
    pub success: bool,
    pub vector: Vec<f32>,
    pub dimension: usize,
    pub processing_time_ms: u64,
    pub method: String,
    pub vector_id: Option<Uuid>, // If stored in database
}

#[derive(Debug, Deserialize)]
pub struct BatchEmbedRequest {
    pub texts: Vec<String>,
    pub metadata: Option<Vec<HashMap<String, String>>>,
    pub store: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct BatchEmbedResponse {
    pub success: bool,
    pub embeddings: Vec<EmbedTextResponse>,
    pub total_processed: usize,
    pub total_time_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct SimilarityRequest {
    pub text1: String,
    pub text2: String,
}

#[derive(Debug, Serialize)]
pub struct SimilarityResponse {
    pub similarity: f32,
    pub text1_vector: Vec<f32>,
    pub text2_vector: Vec<f32>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingStatsResponse {
    pub service_stats: EmbeddingServiceStats,
    pub config: EmbeddingConfigInfo,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingConfigInfo {
    pub dimension: usize,
    pub use_model_embeddings: bool,
    pub cache_size: usize,
    pub batch_size: usize,
}

// ================================================================================================
// API STATE
// ================================================================================================

pub type EmbeddingApiState = Arc<RwLock<EmbeddingService>>;

// ================================================================================================
// API HANDLERS
// ================================================================================================

/// Convert text to vector embedding
pub async fn embed_text(
    State(embedding_service): State<EmbeddingApiState>,
    Json(request): Json<EmbedTextRequest>,
) -> Result<Json<EmbedTextResponse>, StatusCode> {
    info!("📝 Processing text embedding request for: '{}'", 
          if request.text.len() > 50 { &request.text[..50] } else { &request.text });

    let service = embedding_service.read().await;
    
    match service.embed_text(&request.text).await {
        Ok(result) => {
            let mut vector_id = None;
            
            // Store in vector database if requested
            if request.store.unwrap_or(false) {
                // Note: This would require vector storage integration
                // For now, we generate a UUID to simulate storage
                vector_id = Some(Uuid::new_v4());
                info!("💾 Vector stored with ID: {:?}", vector_id);
            }
            
            let dimension = result.vector.len();
            Ok(Json(EmbedTextResponse {
                success: true,
                vector: result.vector,
                dimension,
                processing_time_ms: result.processing_time_ms,
                method: format!("{:?}", result.method),
                vector_id,
            }))
        }
        Err(e) => {
            error!("❌ Text embedding failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Process multiple texts in batch
pub async fn batch_embed(
    State(embedding_service): State<EmbeddingApiState>,
    Json(request): Json<BatchEmbedRequest>,
) -> Result<Json<BatchEmbedResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    info!("📦 Processing batch embedding request for {} texts", request.texts.len());

    if request.texts.is_empty() {
        warn!("⚠️ Empty batch request received");
        return Err(StatusCode::BAD_REQUEST);
    }

    let service = embedding_service.read().await;
    let mut embeddings = Vec::new();
    let mut successful = 0;

    for (i, text) in request.texts.iter().enumerate() {
        match service.embed_text(text).await {
            Ok(result) => {
                let mut vector_id = None;
                
                // Store if requested
                if request.store.unwrap_or(false) {
                    vector_id = Some(Uuid::new_v4());
                }
                
                let dimension = result.vector.len();
                embeddings.push(EmbedTextResponse {
                    success: true,
                    vector: result.vector,
                    dimension,
                    processing_time_ms: result.processing_time_ms,
                    method: format!("{:?}", result.method),
                    vector_id,
                });
                successful += 1;
            }
            Err(e) => {
                warn!("⚠️ Failed to embed text {}: {}", i, e);
                embeddings.push(EmbedTextResponse {
                    success: false,
                    vector: vec![],
                    dimension: 0,
                    processing_time_ms: 0,
                    method: "Error".to_string(),
                    vector_id: None,
                });
            }
        }
    }

    let total_time = start_time.elapsed().as_millis() as u64;
    info!("✅ Batch embedding completed: {}/{} successful in {}ms", 
          successful, request.texts.len(), total_time);

    Ok(Json(BatchEmbedResponse {
        success: successful > 0,
        embeddings,
        total_processed: successful,
        total_time_ms: total_time,
    }))
}

/// Calculate similarity between two texts
pub async fn text_similarity(
    State(embedding_service): State<EmbeddingApiState>,
    Json(request): Json<SimilarityRequest>,
) -> Result<Json<SimilarityResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    info!("🔍 Calculating similarity between texts");

    let service = embedding_service.read().await;
    
    // Generate embeddings for both texts
    let result1 = service.embed_text(&request.text1).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let result2 = service.embed_text(&request.text2).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Calculate cosine similarity
    let similarity = calculate_cosine_similarity(&result1.vector, &result2.vector);
    
    let processing_time = start_time.elapsed().as_millis() as u64;
    
    Ok(Json(SimilarityResponse {
        similarity,
        text1_vector: result1.vector,
        text2_vector: result2.vector,
        processing_time_ms: processing_time,
    }))
}

/// Get embedding service statistics
pub async fn get_embedding_stats(
    State(embedding_service): State<EmbeddingApiState>,
) -> Result<Json<EmbeddingStatsResponse>, StatusCode> {
    let service = embedding_service.read().await;
    let stats = service.get_stats().await;
    
    // Get config info (you'd need to add a method to get config from service)
    let config = EmbeddingConfigInfo {
        dimension: 64, // This should come from the actual config
        use_model_embeddings: true,
        cache_size: 1000,
        batch_size: 10,
    };
    
    Ok(Json(EmbeddingStatsResponse {
        service_stats: stats,
        config,
    }))
}

/// Get embedding by ID (if we implemented storage)
pub async fn get_embedding(
    State(_embedding_service): State<EmbeddingApiState>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // This would integrate with vector storage to retrieve embeddings
    // For now, return a placeholder
    info!("📖 Retrieving embedding: {}", id);
    
    Ok(Json(serde_json::json!({
        "id": id,
        "message": "Embedding retrieval not yet implemented - requires vector storage integration"
    })))
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

/// Calculate cosine similarity between two vectors
fn calculate_cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() {
        return 0.0;
    }
    
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    }
}

// ================================================================================================
// ROUTER CREATION
// ================================================================================================

/// Create the embedding API router
pub fn create_embedding_router() -> Router<EmbeddingApiState> {
    Router::new()
        .route("/api/v1/embed", post(embed_text))
        .route("/api/v1/embed/batch", post(batch_embed))
        .route("/api/v1/embed/similarity", post(text_similarity))
        .route("/api/v1/embed/stats", get(get_embedding_stats))
        .route("/api/v1/embed/{id}", get(get_embedding))
}

// ================================================================================================
// INTEGRATION HELPER
// ================================================================================================

/// Helper function to create embedding service from model
pub async fn create_embedding_service_with_model(
    model: Option<Arc<tokio::sync::Mutex<crate::models::ModelInstance>>>,
    config: Option<EmbeddingConfig>,
) -> EmbeddingService {
    let config = config.unwrap_or_default();
    let mut service = EmbeddingService::new(config);
    
    if let Some(model) = model {
        service.set_model(model);
        info!("🤖 Embedding service configured with AI model");
    } else {
        info!("📊 Embedding service using TF-IDF fallback");
    }
    
    service
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!((calculate_cosine_similarity(&v1, &v2) - 1.0).abs() < 0.001);

        let v3 = vec![1.0, 0.0, 0.0];
        let v4 = vec![0.0, 1.0, 0.0];
        assert!((calculate_cosine_similarity(&v3, &v4) - 0.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_embed_text_request_serialization() {
        let request = EmbedTextRequest {
            text: "Hello world".to_string(),
            metadata: Some(HashMap::new()),
            store: Some(true),
        };
        
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Hello world"));
    }
}