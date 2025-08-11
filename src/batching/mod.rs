use std::{collections::VecDeque, sync::Arc, time::Duration};

use anyhow::Result;
use tokio::{
    sync::{Mutex, oneshot},
    time::Instant,
};
use uuid::Uuid;

#[derive(Debug)]
pub struct BatchRequest {
    pub id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub response_sender: oneshot::Sender<Result<BatchResponse, String>>,
    pub created_at: Instant,
}

#[derive(Debug, Clone)]
pub struct BatchResponse {
    pub text: String,
    pub token_generated: usize,
    pub processing_time_ms: u64,
}

/// Configuration for batching behaviour
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_wait_time_ms: u64,
    pub max_queue_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            max_wait_time_ms: 100,
            max_queue_size: 50,
        }
    }
}

pub struct BatchProcessor {
    config: BatchConfig,
    request_queue: Arc<Mutex<VecDeque<BatchRequest>>>,
    model: Arc<Mutex<crate::models::TinyLlamaModel>>,
    stats: Arc<Mutex<BatchStats>>,
}

#[derive(Debug, Default)]
pub struct BatchStats {
    pub total_requests: u64,
    pub total_batches: u64,
    pub avg_batch_size: f64,
    pub avg_processing_time_ms: f64,
}

impl BatchProcessor {
    pub fn new(config: BatchConfig, model: Arc<Mutex<crate::models::TinyLlamaModel>>) -> Self {
        Self {
            config,
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            model,
            stats: Arc::new(Mutex::new(BatchStats::default())),
        }
    }

    pub async fn submit_request(&self, prompt: String, max_tokens: usize) -> Result<BatchResponse> {
        let (response_tx, response_rx) = oneshot::channel();

        let request = BatchRequest {
            id: Uuid::new_v4().to_string(),
            prompt,
            max_tokens,
            response_sender: response_tx,
            created_at: Instant::now(),
        };

        tracing::debug!("Submitting request for batch queue: {}", request.id);

        // Add to queue
        {
            let mut queue = self.request_queue.lock().await;
            if queue.len() >= self.config.max_queue_size {
                return Err(anyhow::anyhow!(
                    "Request queue is full ({})",
                    self.config.max_queue_size
                ));
            }
            queue.push_back(request);
            tracing::debug!("Queue Size after add {}", queue.len());
        }

        // Wait for response with timeout
        match tokio::time::timeout(Duration::from_secs(30), response_rx).await {
            Ok(Ok(result)) => result.map_err(|e| anyhow::anyhow!(e)),
            Ok(Err(_)) => Err(anyhow::anyhow!("Request was cancelled")),
            Err(_) => Err(anyhow::anyhow!("Request timeout after 30 seconds")),
        }
    }

    /// Start with batch processing loop
    pub async fn start_processing_loop(self: Arc<Self>) {
        tracing::info!(
            "Starting batch processing loop with config: {:?}",
            self.config
        );

        loop {
            let batch = self.collect_batch().await;

            if !batch.is_empty() {
                tracing::debug!("Processing batch of {} requests", batch.len());
                self.process_batch(batch).await;
            } else {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }

    async fn collect_batch(&self) -> Vec<BatchRequest> {
        let mut batch = Vec::new();
        let batch_start = Instant::now();

        loop {
            let request_opt = {
                let mut queue = self.request_queue.lock().await;
                queue.pop_front()
            };

            match request_opt {
                Some(request) => {
                    batch.push(request);

                    if batch.len() >= self.config.max_batch_size {
                        tracing::debug!("Batch fulll with {} request", batch.len());
                        break;
                    }
                }
                None => {
                    if !batch.is_empty() {
                        let wait_time = batch_start.elapsed();
                        if wait_time.as_millis() >= self.config.max_wait_time_ms as u128 {
                            tracing::debug!(
                                "Max wait time reached, processing batch of {}",
                                batch.len()
                            );
                            break;
                        }

                        // Wait a bit for more requests to arrive
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    } else {
                        break;
                    }
                }
            }
        }
        batch
    }

    async fn process_batch(&self, batch: Vec<BatchRequest>) {
        let batch_start = Instant::now();
        let batch_size = batch.len();

        tracing::info!("Processing batch of {} requests", batch_size);

        // For now we are process requests sequentially within the batch
        // TODO: Future Optimization- true parallel batching

        let mut model = self.model.lock().await;

        for request in batch {
            let request_start = Instant::now();

            let result = model.generate(&request.prompt, request.max_tokens).await;

            let processing_time = request_start.elapsed().as_millis() as u64;

            // Prepare Response
            let response = match result {
                Ok(text) => {
                    let token_generated = text.split_whitespace().count();
                    Ok(BatchResponse {
                        text,
                        token_generated,
                        processing_time_ms: processing_time,
                    })
                }
                Err(e) => {
                    tracing::error!("Generation failed for request {}:{}", request.id, e);
                    Err(e.to_string())
                }
            };

            // Send response back through the channel
            if request.response_sender.send(response).is_err() {
                tracing::warn!("Failed to send response for request {}", request.id);
            }
        }

        // Update Statistics
        let total_batch_time = batch_start.elapsed().as_millis() as u64;
        self.update_stats(batch_size, total_batch_time).await;
        tracing::info!(
            "Completed batch of {} requests in {}ms (avg: {}ms per request",
            batch_size,
            total_batch_time,
            total_batch_time / batch_size as u64
        );
    }

    async fn update_stats(&self, batch_size: usize, processing_time_ms: u64) {
        let mut stats = self.stats.lock().await;
        stats.total_requests += batch_size as u64;
        stats.total_batches += 1;
        stats.avg_batch_size = stats.total_requests as f64 / stats.total_batches as f64;

        // Rolling average of processing time
        let alpha = 0.1;
        if stats.avg_processing_time_ms == 0.0 {
            stats.avg_processing_time_ms = processing_time_ms as f64;
        } else {
            stats.avg_processing_time_ms =
                alpha * (processing_time_ms as f64) + (1.0 - alpha) * stats.avg_processing_time_ms;
        }
    }

    pub async fn get_stats(&self) -> BatchStats {
        let stats = self.stats.lock().await;

        BatchStats {
            total_requests: stats.total_requests,
            total_batches: stats.total_batches,
            avg_batch_size: stats.avg_batch_size,
            avg_processing_time_ms: stats.avg_processing_time_ms,
        }
    }

    pub async fn get_queue_size(&self) -> usize {
        let queue = self.request_queue.lock().await;
        queue.len()
    }
}
