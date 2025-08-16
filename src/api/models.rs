use crate::{errors::AppError, models::{ModelVersionManager, AtomicModelSwap, ModelVersion, HealthCheckResult, SwapResult, SwapSafetyReport}};
use axum::{extract::{State, Path}, response::Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub name: String,
    pub version: String,
    pub model_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LoadModelResponse {
    pub model_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct SwapModelRequest {
    pub target_model_id: String,
}

/// Load a new model version
pub async fn load_model(
    State((version_manager, _)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
    Json(request): Json<LoadModelRequest>,
) -> Result<Json<LoadModelResponse>, AppError> {
    tracing::info!("Loading new model: {} v{}", request.name, request.version);
    
    let model_id = version_manager
        .load_model_version(request.name.clone(), request.version.clone(), request.model_path)
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to load model: {}", e)))?;

    Ok(Json(LoadModelResponse {
        model_id,
        status: "loading".to_string(),
        message: format!("Model {} v{} is being loaded", request.name, request.version),
    }))
}

/// List all model versions
pub async fn list_models(
    State((version_manager, _)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
) -> Result<Json<Vec<ModelVersion>>, AppError> {
    let models = version_manager.list_models().await;
    Ok(Json(models))
}

/// Get specific model version info
pub async fn get_model(
    State((version_manager, _)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelVersion>, AppError> {
    let model_version = version_manager
        .get_model_version(&model_id)
        .await
        .ok_or_else(|| AppError::BadRequest(format!("Model {} not found", model_id)))?;

    Ok(Json(model_version))
}

/// Perform health check on a model
pub async fn health_check_model(
    State((version_manager, _)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
    Path(model_id): Path<String>,
) -> Result<Json<HealthCheckResult>, AppError> {
    let health_result = version_manager
        .health_check_model(&model_id)
        .await
        .map_err(|e| AppError::BadRequest(format!("Health check failed: {}", e)))?;

    Ok(Json(health_result))
}

/// Perform atomic model swap
pub async fn swap_model(
    State((_, atomic_swap)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
    Json(request): Json<SwapModelRequest>,
) -> Result<Json<SwapResult>, AppError> {
    tracing::info!("Initiating atomic swap to model: {}", request.target_model_id);
    
    let swap_result = atomic_swap
        .swap_model(&request.target_model_id)
        .await
        .map_err(|e| AppError::BadRequest(format!("Swap failed: {}", e)))?;

    Ok(Json(swap_result))
}

/// Validate if model swap is safe
pub async fn validate_swap_safety(
    State((_, atomic_swap)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
    Path(model_id): Path<String>,
) -> Result<Json<SwapSafetyReport>, AppError> {
    let safety_report = atomic_swap
        .validate_swap_safety(&model_id)
        .await
        .map_err(|e| AppError::BadRequest(format!("Safety validation failed: {}", e)))?;

    Ok(Json(safety_report))
}

/// Rollback to previous model
pub async fn rollback_model(
    State((_, atomic_swap)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
) -> Result<Json<SwapResult>, AppError> {
    tracing::warn!("Initiating rollback operation");
    
    let rollback_result = atomic_swap
        .rollback()
        .await
        .map_err(|e| AppError::BadRequest(format!("Rollback failed: {}", e)))?;

    Ok(Json(rollback_result))
}

/// Get currently active model ID
pub async fn get_active_model(
    State((version_manager, _)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
) -> Result<Json<Option<String>>, AppError> {
    let active_model_id = version_manager.get_active_model_id().await;
    Ok(Json(active_model_id))
}

/// Remove a model version
pub async fn remove_model(
    State((version_manager, _)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    version_manager
        .remove_model(&model_id)
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to remove model: {}", e)))?;

    Ok(Json(serde_json::json!({
        "message": format!("Model {} removed successfully", model_id)
    })))
}

/// Get system status
pub async fn get_system_status(
    State((version_manager, _)): State<(Arc<ModelVersionManager>, Arc<AtomicModelSwap>)>,
) -> Result<Json<serde_json::Value>, AppError> {
    let system_status = version_manager.get_system_status().await;
    
    Ok(Json(serde_json::json!({
        "total_models": system_status.total_models,
        "loaded_models": system_status.loaded_models,
        "active_model_id": system_status.active_model_id,
        "status_counts": system_status.status_counts,
        "max_models": system_status.max_models
    })))
}