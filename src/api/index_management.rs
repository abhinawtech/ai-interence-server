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
             ReindexManager, ReindexJobStatus, ReindexJobState, JobPriority,
             IndexMonitor, PerformanceWindow, AlertSeverity, AlertRule, AlertComparison, 
             ActiveAlert, HealthStatus, CollectionHealth, MetricDataPoint},
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

/// Request to create an alert rule
#[derive(Debug, Deserialize)]
pub struct CreateAlertRuleRequest {
    pub rule_name: String,
    pub metric_name: String,
    pub threshold: f64,
    pub comparison: AlertComparison,
    pub severity: AlertSeverity,
    pub duration_minutes: u64,
    pub collection_pattern: Option<String>,
}

/// Request to record metrics
#[derive(Debug, Deserialize)]
pub struct RecordMetricRequest {
    pub metric_name: String,
    pub value: f64,
    pub tags: Option<HashMap<String, String>>,
}

/// Query parameters for metric history
#[derive(Debug, Deserialize)]
pub struct MetricHistoryQuery {
    pub hours: Option<u64>,
    pub metric: Option<String>,
}

/// Query parameters for performance window
#[derive(Debug, Deserialize)]
pub struct PerformanceWindowQuery {
    pub window_minutes: Option<u64>,
}

/// Application state for index management
pub struct IndexManagementState {
    pub index_optimizer: Arc<IndexOptimizer>,
    pub reindex_manager: Arc<ReindexManager>,
    pub index_monitor: Arc<IndexMonitor>,
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

/// Record a metric for a collection
pub async fn record_metric(
    State(state): State<Arc<IndexManagementState>>,
    Path(collection_name): Path<String>,
    Json(request): Json<RecordMetricRequest>,
) -> Result<Json<OptimizationResponse>, AppError> {
    state.index_monitor.record_metric(
        &collection_name,
        &request.metric_name,
        request.value,
        request.tags,
    ).await;
    
    Ok(Json(OptimizationResponse {
        success: true,
        message: format!("Recorded metric '{}' for collection '{}'", request.metric_name, collection_name),
        performance_metrics: None,
        recommendations: None,
    }))
}

/// Get performance window for a collection
pub async fn get_performance_window(
    State(state): State<Arc<IndexManagementState>>,
    Path(collection_name): Path<String>,
    Query(params): Query<PerformanceWindowQuery>,
) -> Result<Json<Option<PerformanceWindow>>, AppError> {
    let window_minutes = params.window_minutes.unwrap_or(15);
    let window = state.index_monitor.get_performance_window(&collection_name, window_minutes).await;
    Ok(Json(window))
}

/// Get collection health status
pub async fn get_collection_health(
    State(state): State<Arc<IndexManagementState>>,
    Path(collection_name): Path<String>,
) -> Result<Json<Option<CollectionHealth>>, AppError> {
    let health = state.index_monitor.get_collection_health(&collection_name).await;
    Ok(Json(health))
}

/// Get all collection health statuses
pub async fn get_all_health_statuses(
    State(state): State<Arc<IndexManagementState>>,
) -> Result<Json<HashMap<String, CollectionHealth>>, AppError> {
    let health_statuses = state.index_monitor.get_all_health_statuses().await;
    Ok(Json(health_statuses))
}

/// Create an alert rule
pub async fn create_alert_rule(
    State(state): State<Arc<IndexManagementState>>,
    Json(request): Json<CreateAlertRuleRequest>,
) -> Result<Json<OptimizationResponse>, AppError> {
    let rule_id = Uuid::new_v4();
    let rule = AlertRule {
        rule_id,
        rule_name: request.rule_name.clone(),
        metric_name: request.metric_name,
        threshold: request.threshold,
        comparison: request.comparison,
        severity: request.severity,
        duration_minutes: request.duration_minutes,
        enabled: true,
        collection_pattern: request.collection_pattern,
    };
    
    state.index_monitor.create_alert_rule(rule).await?;
    
    Ok(Json(OptimizationResponse {
        success: true,
        message: format!("Created alert rule '{}' with ID: {}", request.rule_name, rule_id),
        performance_metrics: None,
        recommendations: None,
    }))
}

/// Get active alerts
pub async fn get_active_alerts(
    State(state): State<Arc<IndexManagementState>>,
) -> Result<Json<Vec<ActiveAlert>>, AppError> {
    let alerts = state.index_monitor.get_active_alerts().await;
    Ok(Json(alerts))
}

/// Get metric history
pub async fn get_metric_history(
    State(state): State<Arc<IndexManagementState>>,
    Path(collection_name): Path<String>,
    Query(params): Query<MetricHistoryQuery>,
) -> Result<Json<Vec<MetricDataPoint>>, AppError> {
    let hours = params.hours.unwrap_or(24);
    let metric = params.metric.as_deref().unwrap_or("query_latency_ms");
    
    let history = state.index_monitor.get_metric_history(&collection_name, metric, hours).await;
    Ok(Json(history))
}

/// Simulate metrics collection for testing
pub async fn simulate_metrics(
    State(state): State<Arc<IndexManagementState>>,
    Path(collection_name): Path<String>,
) -> Result<Json<OptimizationResponse>, AppError> {
    state.index_monitor.simulate_metrics_collection(&collection_name).await;
    
    Ok(Json(OptimizationResponse {
        success: true,
        message: format!("Simulated metrics collection for collection '{}'", collection_name),
        performance_metrics: None,
        recommendations: None,
    }))
}

/// Resolve an alert
pub async fn resolve_alert(
    State(state): State<Arc<IndexManagementState>>,
    Path(alert_id): Path<Uuid>,
) -> Result<Json<OptimizationResponse>, AppError> {
    state.index_monitor.resolve_alert(alert_id).await?;
    
    Ok(Json(OptimizationResponse {
        success: true,
        message: format!("Resolved alert: {}", alert_id),
        performance_metrics: None,
        recommendations: None,
    }))
}

/// Get all alert rules
pub async fn get_alert_rules(
    State(state): State<Arc<IndexManagementState>>,
) -> Result<Json<HashMap<Uuid, AlertRule>>, AppError> {
    let rules = state.index_monitor.get_alert_rules().await;
    Ok(Json(rules))
}

/// Create index management router
pub fn create_index_management_router(
    index_optimizer: Arc<IndexOptimizer>, 
    reindex_manager: Arc<ReindexManager>,
    index_monitor: Arc<IndexMonitor>
) -> Router {
    let state = Arc::new(IndexManagementState { 
        index_optimizer,
        reindex_manager,
        index_monitor,
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
        
        // Monitoring and metrics endpoints
        .route("/monitor/collections/:collection_name/metrics", post(record_metric))
        .route("/monitor/collections/:collection_name/performance", get(get_performance_window))
        .route("/monitor/collections/:collection_name/health", get(get_collection_health))
        .route("/monitor/collections/:collection_name/metrics/history", get(get_metric_history))
        .route("/monitor/collections/:collection_name/simulate", post(simulate_metrics))
        .route("/monitor/health", get(get_all_health_statuses))
        .route("/monitor/alerts", get(get_active_alerts))
        .route("/monitor/alerts", post(create_alert_rule))
        .route("/monitor/alerts/:alert_id/resolve", post(resolve_alert))
        .route("/monitor/alert-rules", get(get_alert_rules))
        
        .with_state(state)
}