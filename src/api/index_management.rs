use axum::{
    extract::{Path, Query, State},
    response::Json,
    routing::{get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    AppError,
    vector::{IndexOptimizer, IndexProfile, IndexPerformanceMetrics, CollectionIndexConfig},
};

/// Request to register collection for optimization
#[derive(Debug, Deserialize)]
pub struct RegisterCollectionRequest {
    pub collection_name: String,
    pub profile: IndexProfile,
    pub auto_optimize: Option<bool>,
}

/// Request to optimize existing collection
#[derive(Debug, Deserialize)]
pub struct OptimizeCollectionRequest {
    pub profile: IndexProfile,
}

/// Response for collection optimization
#[derive(Debug, Serialize)]
pub struct OptimizationResponse {
    pub success: bool,
    pub message: String,
    pub performance_metrics: Option<IndexPerformanceMetrics>,
    pub recommendations: Option<Vec<String>>,
}

/// Query parameters for benchmark
#[derive(Debug, Deserialize)]
pub struct BenchmarkQuery {
    pub test_queries: Option<usize>,
}

/// Application state for index management
pub struct IndexManagementState {
    pub index_optimizer: Arc<IndexOptimizer>,
}

/// Register collection for optimization
pub async fn register_collection(
    State(state): State<Arc<IndexManagementState>>,
    Json(request): Json<RegisterCollectionRequest>,
) -> Result<Json<OptimizationResponse>, AppError> {
    let auto_optimize = request.auto_optimize.unwrap_or(false);
    
    match state.index_optimizer.register_collection(
        &request.collection_name,
        request.profile.clone(),
        auto_optimize,
    ).await {
        Ok(_) => {
            let metrics = state.index_optimizer
                .analyze_collection_performance(&request.collection_name)
                .await
                .ok();

            Ok(Json(OptimizationResponse {
                success: true,
                message: format!("Registered collection '{}' for optimization with profile: {:?}", 
                               request.collection_name, request.profile),
                performance_metrics: metrics,
                recommendations: None,
            }))
        }
        Err(e) => Err(e),
    }
}

/// Optimize existing collection
pub async fn optimize_collection(
    State(state): State<Arc<IndexManagementState>>,
    Path(collection_name): Path<String>,
    Json(request): Json<OptimizeCollectionRequest>,
) -> Result<Json<OptimizationResponse>, AppError> {
    match state.index_optimizer.update_collection_profile(&collection_name, request.profile.clone()).await {
        Ok(_) => {
            let metrics = state.index_optimizer
                .analyze_collection_performance(&collection_name)
                .await
                .ok();

            let recommendations = state.index_optimizer
                .get_optimization_recommendations(&collection_name)
                .await
                .ok();

            Ok(Json(OptimizationResponse {
                success: true,
                message: format!("Optimized collection '{}' with profile: {:?}", 
                               collection_name, request.profile),
                performance_metrics: metrics,
                recommendations,
            }))
        }
        Err(e) => Err(e),
    }
}

/// Get collection performance analysis
pub async fn analyze_performance(
    State(state): State<Arc<IndexManagementState>>,
    Path(collection_name): Path<String>,
) -> Result<Json<OptimizationResponse>, AppError> {
    let metrics = state.index_optimizer
        .analyze_collection_performance(&collection_name)
        .await?;

    let recommendations = state.index_optimizer
        .get_optimization_recommendations(&collection_name)
        .await?;

    Ok(Json(OptimizationResponse {
        success: true,
        message: format!("Performance analysis for collection '{}'", collection_name),
        performance_metrics: Some(metrics),
        recommendations: Some(recommendations),
    }))
}

/// Auto-optimize collection based on performance
pub async fn auto_optimize_collection(
    State(state): State<Arc<IndexManagementState>>,
    Path(collection_name): Path<String>,
) -> Result<Json<OptimizationResponse>, AppError> {
    match state.index_optimizer.auto_optimize_collection(&collection_name).await {
        Ok(optimized) => {
            let metrics = state.index_optimizer
                .analyze_collection_performance(&collection_name)
                .await
                .ok();

            let message = if optimized {
                format!("Auto-optimized collection '{}'", collection_name)
            } else {
                format!("Collection '{}' is already optimal or auto-optimization is disabled", collection_name)
            };

            Ok(Json(OptimizationResponse {
                success: true,
                message,
                performance_metrics: metrics,
                recommendations: None,
            }))
        }
        Err(e) => Err(e),
    }
}

/// Get all collection configurations
pub async fn get_all_configurations(
    State(state): State<Arc<IndexManagementState>>,
) -> Result<Json<HashMap<String, CollectionIndexConfig>>, AppError> {
    let configs = state.index_optimizer.get_all_configurations().await;
    Ok(Json(configs))
}

/// Benchmark different index configurations
pub async fn benchmark_configurations(
    State(state): State<Arc<IndexManagementState>>,
    Path(collection_name): Path<String>,
    Query(params): Query<BenchmarkQuery>,
) -> Result<Json<HashMap<String, IndexPerformanceMetrics>>, AppError> {
    let test_queries = params.test_queries.unwrap_or(100);
    
    let results = state.index_optimizer
        .benchmark_configurations(&collection_name, test_queries)
        .await?;

    Ok(Json(results))
}

/// Create index management router
pub fn create_index_management_router(index_optimizer: Arc<IndexOptimizer>) -> Router {
    let state = Arc::new(IndexManagementState { index_optimizer });

    Router::new()
        .route("/collections/register", post(register_collection))
        .route("/collections/:collection_name/optimize", put(optimize_collection))
        .route("/collections/:collection_name/analyze", get(analyze_performance))
        .route("/collections/:collection_name/auto-optimize", post(auto_optimize_collection))
        .route("/collections/:collection_name/benchmark", get(benchmark_configurations))
        .route("/configurations", get(get_all_configurations))
        .with_state(state)
}