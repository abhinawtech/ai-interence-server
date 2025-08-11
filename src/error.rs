use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model loading error: {0}")]
    ModelLoading(#[from] anyhow::Error),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Internal server error")]
    Internal,
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            Error::ModelNotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            Error::ModelLoading(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            Error::Tokenization(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            Error::Generation(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            Error::Config(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            Error::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "IO error".to_string()),
            Error::Serialization(_) => (StatusCode::BAD_REQUEST, "Invalid JSON".to_string()),
            Error::Internal => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".to_string(),
            ),
        };

        let body = Json(json!({
            "error": error_message,
            "status": status.as_u16()
        }));

        (status, body).into_response()
    }
}
