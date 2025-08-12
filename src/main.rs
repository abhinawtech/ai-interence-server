use ai_interence_server::api::generate::generate_text;
use ai_interence_server::api::health::health_check;
use ai_interence_server::batching::{BatchConfig, BatchProcessor};
use ai_interence_server::models::TinyLlamaModel;
use axum::{
    Router,
    routing::{get, post},
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Set optimal thread count for M1
    std::env::set_var("RAYON_NUM_THREADS", "4");  // M1 has 4 performance cores
    std::env::set_var("TOKENIZERS_PARALLELISM", "false");

    // Load environment variables from .env file
    dotenv::dotenv().ok();

    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();

    tracing::info!("🚀 Starting AI Inference Server with Batching");

    // Load batch configuration
    let batch_config = BatchConfig {
        max_batch_size: std::env::var("BATCH_MAX_SIZE")
            .unwrap_or_else(|_| "4".to_string())
            .parse()
            .unwrap_or(4),
        max_wait_time_ms: std::env::var("BATCH_MAX_WAIT_MS")
            .unwrap_or_else(|_| "100".to_string())
            .parse()
            .unwrap_or(100),
        max_queue_size: std::env::var("BATCH_MAX_QUEUE_SIZE")
            .unwrap_or_else(|_| "50".to_string())
            .parse()
            .unwrap_or(50),
    };

    tracing::info!(
        "📊 Batch Configuration - Max Size: {}, Max Wait: {}ms, Max Queue: {}",
        batch_config.max_batch_size,
        batch_config.max_wait_time_ms,
        batch_config.max_queue_size
    );

    // Initialize the TinyLlamaModel 
    tracing::info!("🚀 Loading TinyLlama model...");
    let mut model = TinyLlamaModel::load().await?;
    
    // Warm up the model with a sample generation
    tracing::info!("🔥 Warming up model with sample generation...");
    let warmup_start = std::time::Instant::now();
    let _warmup_result = model.generate("Hello", 5).await;
    let warmup_time = warmup_start.elapsed();
    tracing::info!("✅ Model warmed up in {:?}", warmup_time);
    
    let shared_model = Arc::new(Mutex::new(model));
    tracing::info!("✅ TinyLlama model loaded successfully");

    // Initialize BatchProcessor
    tracing::info!("🔄 Initializing BatchProcessor...");
    let batch_processor = Arc::new(BatchProcessor::new(batch_config.clone(), Arc::clone(&shared_model)));
    tracing::info!("✅ BatchProcessor initialized");

    // Start the batch processing loop in background task
    tracing::info!("🎯 Starting batch processing loop...");
    let processor_clone = Arc::clone(&batch_processor);
    let batch_task = tokio::spawn(async move {
        processor_clone.start_processing_loop().await;
    });

    // Create the router with BatchProcessor state
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/generate", post(generate_text))
        .with_state(Arc::clone(&batch_processor));

    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse()
        .unwrap_or(3000);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("🌐 Server starting on http://{}", addr);
    tracing::info!("📡 Available endpoints:");
    tracing::info!("  • GET  /health - Health check");
    tracing::info!("  • POST /api/v1/generate - Text generation");
    tracing::info!("  • GET  /api/v1/batch/status - Batch processing status");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let server = axum::serve(listener, app);

    tracing::info!("✅ Server ready and accepting requests");

    // Setup graceful shutdown
    tokio::select! {
        result = server => {
            if let Err(e) = result {
                tracing::error!("❌ Server error: {}", e);
                return Err(e.into());
            }
        }
        _ = shutdown_signal() => {
            tracing::info!("🛑 Shutdown signal received");
        }
    }

    // Clean shutdown of batch task
    batch_task.abort();
    tracing::info!("👋 Server shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
