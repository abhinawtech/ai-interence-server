// ================================================================================================
// SIMPLE VECTOR API
// ================================================================================================
//
// Basic REST API for vector operations using in-memory storage.
// This is a minimal implementation to get started.
//
// ================================================================================================

use crate::vector::{VectorPoint, VectorBackend, VectorResult};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

// ================================================================================================
// API TYPES
// ================================================================================================

#[derive(Debug, Deserialize)]
pub struct InsertVectorRequest {
    pub vector: Vec<f32>,
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct InsertVectorResponse {
    pub id: Uuid,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchVectorRequest {
    pub vector: Vec<f32>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct SearchVectorResponse {
    pub results: Vec<SearchResult>,
    pub total_found: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: Uuid,
    pub similarity: f32,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct VectorStatsResponse {
    pub total_vectors: usize,
    pub memory_usage_estimate: usize,
}

// ================================================================================================
// API STATE
// ================================================================================================

pub type VectorApiState = Arc<VectorBackend>;

// ================================================================================================
// API HANDLERS
// ================================================================================================

/// Insert a new vector
pub async fn insert_vector(
    State(backend): State<VectorApiState>,
    Json(request): Json<InsertVectorRequest>,
) -> Result<Json<InsertVectorResponse>, StatusCode> {
    let metadata = request.metadata.unwrap_or_default();
    let point = VectorPoint::with_metadata(request.vector, metadata);
    let id = point.id;
    
    match backend.insert(point).await {
        Ok(_) => Ok(Json(InsertVectorResponse {
            id,
            success: true,
            message: "Vector inserted successfully".to_string(),
        })),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Search for similar vectors
pub async fn search_vectors(
    State(backend): State<VectorApiState>,
    Json(request): Json<SearchVectorRequest>,
) -> Result<Json<SearchVectorResponse>, StatusCode> {
    let limit = request.limit.unwrap_or(10);
    
    match backend.search_similar(&request.vector, limit).await {
        Ok(results) => {
            let mut search_results = Vec::new();
            for (id, similarity) in results {
                if let Some(point) = backend.get(&id).await {
                    search_results.push(SearchResult {
                        id,
                        similarity,
                        metadata: point.metadata.clone(),
                    });
                }
            }
            
            Ok(Json(SearchVectorResponse {
                total_found: search_results.len(),
                results: search_results,
            }))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Get vector by ID
pub async fn get_vector(
    State(backend): State<VectorApiState>,
    Path(id): Path<Uuid>,
) -> Result<Json<VectorPoint>, StatusCode> {
    match backend.get(&id).await {
        Some(point) => Ok(Json(point)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// Delete vector by ID
pub async fn delete_vector(
    State(backend): State<VectorApiState>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match backend.delete(&id).await {
        Ok(true) => Ok(Json(serde_json::json!({
            "success": true,
            "message": "Vector deleted successfully"
        }))),
        Ok(false) => Err(StatusCode::NOT_FOUND),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Get storage statistics
pub async fn get_stats(
    State(backend): State<VectorApiState>,
) -> Result<Json<VectorStatsResponse>, StatusCode> {
    let stats = backend.stats().await;
    Ok(Json(VectorStatsResponse {
        total_vectors: stats.total_vectors,
        memory_usage_estimate: stats.memory_usage_estimate,
    }))
}

/// List all vectors (for debugging)
pub async fn list_vectors(
    State(backend): State<VectorApiState>,
) -> Result<Json<Vec<VectorPoint>>, StatusCode> {
    let all_vectors = backend.list_all().await;
    Ok(Json(all_vectors))
}

// ================================================================================================
// ROUTER CREATION
// ================================================================================================

/// Create the vector API router
pub fn create_vector_router() -> Router<VectorApiState> {
    Router::new()
        .route("/api/v1/vectors", post(insert_vector))
        .route("/api/v1/vectors/search", post(search_vectors))
        .route("/api/v1/vectors/stats", get(get_stats))
        .route("/api/v1/vectors/list", get(list_vectors))
        .route("/api/v1/vectors/{id}", get(get_vector).delete(delete_vector))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use std::collections::HashMap;

    use crate::vector::VectorStorageFactory;
    
    async fn create_test_backend() -> VectorApiState {
        Arc::new(VectorStorageFactory::create_in_memory_backend())
    }

    #[tokio::test]
    async fn test_insert_vector_handler() {
        let backend = create_test_backend().await;
        let request = InsertVectorRequest {
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        };

        let result = insert_vector(State(backend), Json(request)).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert!(response.0.success);
    }

    #[tokio::test]
    async fn test_search_vectors_handler() {
        let backend = create_test_backend().await;
        
        // Insert a test vector first
        let insert_request = InsertVectorRequest {
            vector: vec![1.0, 0.0, 0.0],
            metadata: None,
        };
        let _ = insert_vector(State(backend.clone()), Json(insert_request)).await;

        // Search for similar vectors
        let search_request = SearchVectorRequest {
            vector: vec![1.0, 0.0, 0.0],
            limit: Some(5),
        };

        let result = search_vectors(State(backend), Json(search_request)).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert_eq!(response.0.total_found, 1);
    }
}