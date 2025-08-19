// Quick health API test without full server startup
use axum::{routing::get, Router, response::Json};
use serde_json::json;
use std::net::SocketAddr;
use tokio::signal;

async fn health_check() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "service": "ai-inference-server",
        "version": "0.1.0",
        "architecture": "new-trait-based",
        "models": {
            "registered": 1,
            "available": ["gemma-2b-it", "gemma", "gemma-2b", "gemma2b"],
            "registry_status": "active"
        }
    }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .init();

    println!("🚀 Starting Health API Test Server");
    
    // Simple router with just health endpoint
    let app = Router::new()
        .route("/health", get(health_check));

    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse()
        .unwrap_or(3000);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    
    println!("🌐 Health API server running on http://{}", addr);
    println!("📡 Test with: curl http://localhost:3000/health");
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let server = axum::serve(listener, app);

    println!("✅ Server ready - health API accepting requests");

    // Run server with graceful shutdown
    tokio::select! {
        result = server => {
            if let Err(e) = result {
                println!("❌ Server error: {}", e);
                return Err(e.into());
            }
        }
        _ = signal::ctrl_c() => {
            println!("🛑 Shutdown signal received");
        }
    }

    println!("👋 Server shutdown complete");
    Ok(())
}