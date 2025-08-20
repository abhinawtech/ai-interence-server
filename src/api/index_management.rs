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
    vector::{IndexOptimizer, IndexProfile, IndexPerformanceMetrics, CollectionIndexConfig, 
             ReindexManager, ReindexJobStatus, ReindexJobState, JobPriority},
};
use uuid::Uuid;

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

/// Request to schedule a reindexing job
#[derive(Debug, Deserialize)]
pub struct ScheduleReindexRequest {
    pub collection_name: String,
    pub target_profile: IndexProfile,
    pub priority: Option<JobPriority>,
}

/// Response for job operations
#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub success: bool,
    pub message: String,
    pub job_id: Option<Uuid>,
    pub job_state: Option<ReindexJobState>,
}

/// Queue status response
#[derive(Debug, Serialize)]
pub struct QueueStatusResponse {
    pub queued_jobs: usize,
    pub running_jobs: usize,
    pub total_jobs: usize,
    pub max_concurrent: usize,
}

/// Application state for index management
pub struct IndexManagementState {
    pub index_optimizer: Arc<IndexOptimizer>,
    pub reindex_manager: Arc<ReindexManager>,
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

/// Schedule a background reindexing job
pub async fn schedule_reindex_job(
    State(state): State<Arc<IndexManagementState>>,
    Json(request): Json<ScheduleReindexRequest>,
) -> Result<Json<JobResponse>, AppError> {
    let priority = request.priority.unwrap_or(JobPriority::Medium);
    
    match state.reindex_manager.schedule_reindex_job(
        request.collection_name.clone(),
        request.target_profile.clone(),
        priority,
    ).await {
        Ok(job_id) => {
            // Auto-start the job
            let _ = state.reindex_manager.start_job(job_id).await;
            
            let job_state = state.reindex_manager.get_job_status(job_id).await;
            
            Ok(Json(JobResponse {
                success: true,
                message: format!("Scheduled reindex job for collection '{}'", request.collection_name),
                job_id: Some(job_id),
                job_state,
            }))
        }
        Err(e) => Err(e),
    }
}

/// Get status of a specific job
pub async fn get_job_status(
    State(state): State<Arc<IndexManagementState>>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<JobResponse>, AppError> {
    match state.reindex_manager.get_job_status(job_id).await {
        Some(job_state) => Ok(Json(JobResponse {
            success: true,
            message: format!("Job status: {:?}", job_state.status),
            job_id: Some(job_id),
            job_state: Some(job_state),
        })),
        None => Err(AppError::NotFound(format!("Job {} not found", job_id))),
    }
}

/// Get all jobs
pub async fn get_all_jobs(
    State(state): State<Arc<IndexManagementState>>,
) -> Result<Json<HashMap<Uuid, ReindexJobState>>, AppError> {
    let jobs = state.reindex_manager.get_all_jobs().await;
    Ok(Json(jobs))
}

/// Get jobs by status
pub async fn get_jobs_by_status(
    State(state): State<Arc<IndexManagementState>>,
    Path(status_str): Path<String>,
) -> Result<Json<Vec<ReindexJobState>>, AppError> {
    let status = match status_str.to_lowercase().as_str() {
        "queued" => ReindexJobStatus::Queued,
        "running" => ReindexJobStatus::Running,
        "completed" => ReindexJobStatus::Completed,
        "failed" => ReindexJobStatus::Failed,
        "cancelled" => ReindexJobStatus::Cancelled,
        "paused" => ReindexJobStatus::Paused,
        _ => return Err(AppError::BadRequest("Invalid status".to_string())),
    };
    
    let jobs = state.reindex_manager.get_jobs_by_status(status).await;
    Ok(Json(jobs))
}

/// Get queue status
pub async fn get_queue_status(
    State(state): State<Arc<IndexManagementState>>,
) -> Result<Json<QueueStatusResponse>, AppError> {
    let (queued, running, total) = state.reindex_manager.get_queue_status().await;
    
    Ok(Json(QueueStatusResponse {
        queued_jobs: queued,
        running_jobs: running,
        total_jobs: total,
        max_concurrent: 4, // This would come from configuration
    }))
}

/// Pause a job
pub async fn pause_job(
    State(state): State<Arc<IndexManagementState>>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<JobResponse>, AppError> {
    state.reindex_manager.pause_job_by_id(job_id).await?;
    
    let job_state = state.reindex_manager.get_job_status(job_id).await;
    
    Ok(Json(JobResponse {
        success: true,
        message: format!("Job {} paused", job_id),
        job_id: Some(job_id),
        job_state,
    }))
}

/// Resume a job
pub async fn resume_job(
    State(state): State<Arc<IndexManagementState>>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<JobResponse>, AppError> {
    state.reindex_manager.resume_job_by_id(job_id).await?;
    
    let job_state = state.reindex_manager.get_job_status(job_id).await;
    
    Ok(Json(JobResponse {
        success: true,
        message: format!("Job {} resumed", job_id),
        job_id: Some(job_id),
        job_state,
    }))
}

/// Cancel a job
pub async fn cancel_job(
    State(state): State<Arc<IndexManagementState>>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<JobResponse>, AppError> {
    state.reindex_manager.cancel_job_by_id(job_id).await?;
    
    let job_state = state.reindex_manager.get_job_status(job_id).await;
    
    Ok(Json(JobResponse {
        success: true,
        message: format!("Job {} cancelled", job_id),
        job_id: Some(job_id),
        job_state,
    }))
}

/// Create index management router
pub fn create_index_management_router(
    index_optimizer: Arc<IndexOptimizer>, 
    reindex_manager: Arc<ReindexManager>
) -> Router {
    let state = Arc::new(IndexManagementState { 
        index_optimizer,
        reindex_manager,
    });

    Router::new()
        // Collection optimization endpoints
        .route("/collections/register", post(register_collection))
        .route("/collections/:collection_name/optimize", put(optimize_collection))
        .route("/collections/:collection_name/analyze", get(analyze_performance))
        .route("/collections/:collection_name/auto-optimize", post(auto_optimize_collection))
        .route("/collections/:collection_name/benchmark", get(benchmark_configurations))
        .route("/configurations", get(get_all_configurations))
        
        // Background reindexing endpoints
        .route("/reindex/schedule", post(schedule_reindex_job))
        .route("/reindex/jobs", get(get_all_jobs))
        .route("/reindex/jobs/:job_id", get(get_job_status))
        .route("/reindex/jobs/:job_id/pause", post(pause_job))
        .route("/reindex/jobs/:job_id/resume", post(resume_job))
        .route("/reindex/jobs/:job_id/cancel", post(cancel_job))
        .route("/reindex/jobs/status/:status", get(get_jobs_by_status))
        .route("/reindex/queue/status", get(get_queue_status))
        
        .with_state(state)
}