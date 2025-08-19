// ================================================================================================
// QDRANT CLIENT - PRODUCTION READY CONNECTION MANAGEMENT
// ================================================================================================
//
// High-performance Qdrant client with:
// - Connection pooling and management
// - Automatic retry logic with exponential backoff
// - Health check integration
// - Error handling and recovery
// - Async/await support for non-blocking operations
//
// ================================================================================================

use crate::vector::qdrant_config::QdrantConfig;
use anyhow::Result;
use qdrant_client::{
    Qdrant,
    qdrant::{
        CreateCollection, Distance, VectorParams,
        HealthCheckReply,
    },
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

/// Production-ready Qdrant client wrapper
#[derive(Clone)]
pub struct QdrantClient {
    client: Arc<Qdrant>,
    config: QdrantConfig,
    connection_state: Arc<RwLock<ConnectionState>>,
}

/// Connection state tracking
#[derive(Debug, Clone)]
struct ConnectionState {
    is_healthy: bool,
    last_health_check: std::time::Instant,
    consecutive_failures: u32,
    last_error: Option<String>,
}

impl Default for ConnectionState {
    fn default() -> Self {
        Self {
            is_healthy: false,
            last_health_check: std::time::Instant::now(),
            consecutive_failures: 0,
            last_error: None,
        }
    }
}

impl QdrantClient {
    /// Create a new Qdrant client with connection pooling
    pub async fn new(config: QdrantConfig) -> Result<Self> {
        info!("🔗 Initializing Qdrant client: {}", config.url);
        
        // Validate configuration
        config.validate()
            .map_err(|e| anyhow::anyhow!("Invalid Qdrant configuration: {}", e))?;

        // Create the core Qdrant client with compatibility check disabled
        let client = if let Some(api_key) = &config.api_key {
            let mut client_config = Qdrant::from_url(&config.url)
                .api_key(api_key.clone());
            client_config.check_compatibility = false;  // Disable version compatibility check
            client_config
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to create authenticated Qdrant client: {}", e))?
        } else {
            let mut client_config = Qdrant::from_url(&config.url);
            client_config.check_compatibility = false;  // Disable version compatibility check
            client_config
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to create Qdrant client: {}", e))?
        };

        let qdrant_client = Self {
            client: Arc::new(client),
            config,
            connection_state: Arc::new(RwLock::new(ConnectionState::default())),
        };

        // Initial health check
        match qdrant_client.health_check().await {
            Ok(_) => {
                info!("✅ Qdrant client initialized successfully");
                let mut state = qdrant_client.connection_state.write().await;
                state.is_healthy = true;
                state.last_health_check = std::time::Instant::now();
            }
            Err(e) => {
                warn!("⚠️ Initial Qdrant health check failed: {}", e);
                let mut state = qdrant_client.connection_state.write().await;
                state.is_healthy = false;
                state.last_error = Some(e.to_string());
            }
        }

        Ok(qdrant_client)
    }

    /// Perform health check on Qdrant server
    pub async fn health_check(&self) -> Result<HealthCheckReply> {
        debug!("🔍 Performing Qdrant health check");
        
        let result = tokio::time::timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.client.health_check()
        ).await;

        match result {
            Ok(Ok(health_reply)) => {
                debug!("✅ Qdrant health check successful");
                self.update_connection_state(true, None).await;
                Ok(health_reply)
            }
            Ok(Err(e)) => {
                warn!("❌ Qdrant health check failed: {}", e);
                self.update_connection_state(false, Some(e.to_string())).await;
                Err(e.into())
            }
            Err(_) => {
                let error_msg = format!("Qdrant health check timeout after {}s", self.config.timeout_secs);
                warn!("⏰ {}", error_msg);
                self.update_connection_state(false, Some(error_msg.clone())).await;
                Err(anyhow::anyhow!(error_msg))
            }
        }
    }

    /// Update connection state tracking
    async fn update_connection_state(&self, is_healthy: bool, error: Option<String>) {
        let mut state = self.connection_state.write().await;
        
        if is_healthy {
            state.is_healthy = true;
            state.consecutive_failures = 0;
            state.last_error = None;
        } else {
            state.is_healthy = false;
            state.consecutive_failures += 1;
            if let Some(err) = error {
                state.last_error = Some(err);
            }
        }
        
        state.last_health_check = std::time::Instant::now();
    }

    /// Get current connection state
    pub async fn get_connection_state(&self) -> (bool, u32, Option<String>) {
        let state = self.connection_state.read().await;
        (state.is_healthy, state.consecutive_failures, state.last_error.clone())
    }

    /// Create a collection if it doesn't exist
    pub async fn ensure_collection(&self, collection_name: &str) -> Result<()> {
        info!("📂 Ensuring collection exists: {}", collection_name);

        // Check if collection exists
        match self.client.collection_info(collection_name).await {
            Ok(_) => {
                debug!("✅ Collection '{}' already exists", collection_name);
                return Ok(());
            }
            Err(_) => {
                info!("📁 Creating new collection: {}", collection_name);
            }
        }

        // Create collection with vector configuration
        let create_collection = CreateCollection {
            collection_name: collection_name.to_string(),
            vectors_config: Some(VectorParams {
                size: self.config.vector_dimension as u64,
                distance: Distance::Cosine.into(),
                ..Default::default()
            }.into()),
            ..Default::default()
        };

        let result = self.client.create_collection(create_collection).await;

        match result {
            Ok(_) => {
                info!("✅ Collection '{}' created successfully", collection_name);
                Ok(())
            }
            Err(e) => {
                error!("❌ Failed to create collection '{}': {}", collection_name, e);
                Err(e.into())
            }
        }
    }

    /// Execute operation with retry logic
    async fn execute_with_retry<F, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>> + Send + Sync,
        T: Send,
    {
        let max_retries = 3;
        let mut retry_count = 0;
        let mut last_error = None;

        while retry_count <= max_retries {
            match operation().await {
                Ok(result) => {
                    if retry_count > 0 {
                        info!("✅ Operation succeeded after {} retries", retry_count);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    retry_count += 1;
                    last_error = Some(e);

                    if retry_count <= max_retries {
                        let delay = Duration::from_millis(100 * (1 << retry_count)); // Exponential backoff
                        warn!("⚠️ Operation failed (attempt {}/{}), retrying in {:?}...", 
                              retry_count, max_retries + 1, delay);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        let error = last_error.unwrap();
        error!("❌ Operation failed after {} retries: {}", max_retries + 1, error);
        Err(error)
    }

    /// Get the underlying Qdrant client for direct operations
    pub fn get_client(&self) -> &Qdrant {
        &self.client
    }

    /// Get configuration
    pub fn get_config(&self) -> &QdrantConfig {
        &self.config
    }
}

/// Builder pattern for creating QdrantClient with custom settings
pub struct QdrantClientBuilder {
    config: QdrantConfig,
}

impl QdrantClientBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: QdrantConfig::default(),
        }
    }

    /// Create builder from existing configuration
    pub fn from_config(config: QdrantConfig) -> Self {
        Self { config }
    }

    /// Set Qdrant server URL
    pub fn url<S: Into<String>>(mut self, url: S) -> Self {
        self.config.url = url.into();
        self
    }

    /// Set API key for authentication
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config.api_key = Some(api_key.into());
        self
    }

    /// Set connection timeout
    pub fn timeout(mut self, timeout_secs: u64) -> Self {
        self.config.timeout_secs = timeout_secs;
        self
    }

    /// Set vector dimension
    pub fn vector_dimension(mut self, dimension: usize) -> Self {
        self.config.vector_dimension = dimension;
        self
    }

    /// Build the QdrantClient
    pub async fn build(self) -> Result<QdrantClient> {
        QdrantClient::new(self.config).await
    }
}

impl Default for QdrantClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        let builder = QdrantClientBuilder::new()
            .url("http://localhost:6334")
            .timeout(30)
            .vector_dimension(128);
        
        assert_eq!(builder.config.url, "http://localhost:6334");
        assert_eq!(builder.config.timeout_secs, 30);
        assert_eq!(builder.config.vector_dimension, 128);
    }

    #[test]
    fn test_connection_state_default() {
        let state = ConnectionState::default();
        assert!(!state.is_healthy);
        assert_eq!(state.consecutive_failures, 0);
        assert!(state.last_error.is_none());
    }

    #[tokio::test]
    async fn test_client_creation_with_invalid_config() {
        let mut config = QdrantConfig::default();
        config.url = "".to_string(); // Invalid URL

        let result = QdrantClient::new(config).await;
        assert!(result.is_err());
    }
}