use axum::{routing::{get, post}, Router};
use ai_interence_server::api::{health, generate};
use ai_interence_server::models::TinyLlamaModel;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Initialize the TinyLlamaModel 
    tracing::info!("Loading TinyLlama model...");
    let model = TinyLlamaModel::load().await?;
    let shared_model = Arc::new(tokio::sync::Mutex::new(model));
    tracing::info!("TinyLlama model loaded successfully");

    let app = Router::new()
        .route("/health", get(health::health_check))
        .route("/generate", post(generate::generate_text))
        .with_state(shared_model);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
