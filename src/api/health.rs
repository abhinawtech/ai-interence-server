use axum::{http::StatusCode, response::Json};
use serde_json::{json, Value};

pub async fn health_check() -> Result<(StatusCode, Json<Value>), (StatusCode, Json<Value>)> {
    let response = json!({
        "status": "healthy",
        "service": "ai-inference-server",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339()
    });

    Ok((StatusCode::OK, Json(response)))
}