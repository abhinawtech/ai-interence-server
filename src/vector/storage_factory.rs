// ================================================================================================
// VECTOR STORAGE FACTORY - SMART STORAGE SELECTION
// ================================================================================================
//
// Intelligent storage backend selection with fallback support:
// - Primary: Qdrant for production scalability
// - Fallback: In-memory for development/testing
// - Environment-based configuration
// - Health check integration
// - Seamless switching between backends
//
// ================================================================================================

use crate::vector::{
    VectorStorage, QdrantVectorStorage, QdrantClient, QdrantConfig, QdrantClientBuilder,
    VectorError, VectorResult, VectorPoint, StorageStats,
};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use uuid::Uuid;

/// Unified vector storage backend that can use either Qdrant or in-memory storage
pub enum VectorBackend {
    /// Production Qdrant backend
    Qdrant(QdrantVectorStorage),
    /// Development in-memory backend
    InMemory(Arc<std::sync::Mutex<VectorStorage>>),
}

/// Vector storage factory with intelligent backend selection
pub struct VectorStorageFactory;

impl VectorStorageFactory {
    /// Create the best available vector storage backend
    pub async fn create() -> Result<VectorBackend, VectorError> {
        info!("🏭 Initializing vector storage factory...");
        
        // Try to create Qdrant backend first
        match Self::try_create_qdrant().await {
            Ok(qdrant_storage) => {
                info!("✅ Using Qdrant vector storage backend");
                Ok(VectorBackend::Qdrant(qdrant_storage))
            }
            Err(e) => {
                warn!("⚠️ Qdrant unavailable ({}), falling back to in-memory storage", e);
                let in_memory_storage = Self::create_in_memory();
                info!("✅ Using in-memory vector storage backend");
                Ok(VectorBackend::InMemory(in_memory_storage))
            }
        }
    }
    
    /// Force creation of Qdrant backend (for production)
    pub async fn create_qdrant() -> Result<VectorBackend, VectorError> {
        info!("🗂️ Creating Qdrant vector storage backend...");
        let qdrant_storage = Self::try_create_qdrant().await?;
        info!("✅ Qdrant vector storage backend created successfully");
        Ok(VectorBackend::Qdrant(qdrant_storage))
    }
    
    /// Force creation of in-memory backend (for testing)
    pub fn create_in_memory_backend() -> VectorBackend {
        info!("💾 Creating in-memory vector storage backend...");
        let in_memory_storage = Self::create_in_memory();
        info!("✅ In-memory vector storage backend created");
        VectorBackend::InMemory(in_memory_storage)
    }
    
    /// Try to create Qdrant backend with configuration from environment
    async fn try_create_qdrant() -> Result<QdrantVectorStorage, VectorError> {
        // Load configuration from environment
        let config = QdrantConfig::from_env();
        
        debug!("🔧 Qdrant configuration: URL={}, Collection={}", 
               config.url, config.conversation_collection);
        
        // Validate configuration
        config.validate()
            .map_err(|e| VectorError::Operation(format!("Invalid Qdrant config: {}", e)))?;
        
        // Create Qdrant client
        let client = QdrantClientBuilder::from_config(config)
            .build()
            .await
            .map_err(|e| VectorError::Operation(format!("Failed to create Qdrant client: {}", e)))?;
        
        // Test connectivity with health check
        client.health_check()
            .await
            .map_err(|e| VectorError::Operation(format!("Qdrant health check failed: {}", e)))?;
        
        // Create Qdrant vector storage
        QdrantVectorStorage::new(client)
            .await
            .map_err(|e| VectorError::Operation(format!("Failed to create Qdrant storage: {}", e)))
    }
    
    /// Create in-memory storage backend
    fn create_in_memory() -> Arc<std::sync::Mutex<VectorStorage>> {
        Arc::new(std::sync::Mutex::new(VectorStorage::new()))
    }
}

/// Unified interface for vector operations regardless of backend
impl VectorBackend {
    /// Insert a vector point
    pub async fn insert(&self, point: VectorPoint) -> VectorResult<Uuid> {
        match self {
            VectorBackend::Qdrant(storage) => storage.insert(point).await,
            VectorBackend::InMemory(storage) => {
                let mut storage = storage.lock()
                    .map_err(|e| VectorError::Operation(format!("Lock error: {}", e)))?;
                storage.insert(point)
            }
        }
    }
    
    /// Get a vector point by ID
    pub async fn get(&self, id: &Uuid) -> Option<VectorPoint> {
        match self {
            VectorBackend::Qdrant(storage) => storage.get(id).await,
            VectorBackend::InMemory(storage) => {
                let storage = storage.lock().ok()?;
                storage.get(id).cloned()
            }
        }
    }
    
    /// Delete a vector point
    pub async fn delete(&self, id: &Uuid) -> VectorResult<bool> {
        match self {
            VectorBackend::Qdrant(storage) => storage.delete(id).await,
            VectorBackend::InMemory(storage) => {
                let mut storage = storage.lock()
                    .map_err(|e| VectorError::Operation(format!("Lock error: {}", e)))?;
                storage.delete(id)
            }
        }
    }
    
    /// Search for similar vectors
    pub async fn search_similar(&self, query_vector: &[f32], limit: usize) -> VectorResult<Vec<(Uuid, f32)>> {
        match self {
            VectorBackend::Qdrant(storage) => storage.search_similar(query_vector, limit).await,
            VectorBackend::InMemory(storage) => {
                let storage = storage.lock()
                    .map_err(|e| VectorError::Operation(format!("Lock error: {}", e)))?;
                storage.search_similar(query_vector, limit)
            }
        }
    }
    
    /// List all vectors
    pub async fn list_all(&self) -> Vec<VectorPoint> {
        match self {
            VectorBackend::Qdrant(storage) => storage.list_all().await,
            VectorBackend::InMemory(storage) => {
                if let Ok(storage) = storage.lock() {
                    storage.list_all().into_iter().cloned().collect()
                } else {
                    Vec::new()
                }
            }
        }
    }
    
    /// Get storage statistics
    pub async fn stats(&self) -> StorageStats {
        match self {
            VectorBackend::Qdrant(storage) => storage.stats().await,
            VectorBackend::InMemory(storage) => {
                if let Ok(storage) = storage.lock() {
                    storage.stats()
                } else {
                    StorageStats {
                        total_vectors: 0,
                        memory_usage_estimate: 0,
                    }
                }
            }
        }
    }
    
    /// Batch insert vectors (Qdrant optimization)
    pub async fn batch_insert(&self, points: Vec<VectorPoint>) -> VectorResult<Vec<Uuid>> {
        match self {
            VectorBackend::Qdrant(storage) => storage.batch_insert(points).await,
            VectorBackend::InMemory(storage) => {
                // For in-memory, fall back to individual inserts
                let mut ids = Vec::new();
                for point in points {
                    let id = self.insert(point).await?;
                    ids.push(id);
                }
                Ok(ids)
            }
        }
    }
    
    /// Get backend type for debugging
    pub fn backend_type(&self) -> &'static str {
        match self {
            VectorBackend::Qdrant(_) => "qdrant",
            VectorBackend::InMemory(_) => "in-memory",
        }
    }
    
    /// Check if backend is production-ready
    pub fn is_production_ready(&self) -> bool {
        matches!(self, VectorBackend::Qdrant(_))
    }
    
    /// Get enhanced statistics (Qdrant-specific)
    pub async fn enhanced_stats(&self) -> Option<crate::vector::QdrantStorageStats> {
        match self {
            VectorBackend::Qdrant(storage) => Some(storage.qdrant_stats().await),
            VectorBackend::InMemory(_) => None,
        }
    }
}

/// Configuration for vector storage factory
#[derive(Debug, Clone)]
pub struct VectorStorageConfig {
    /// Prefer Qdrant over in-memory
    pub prefer_qdrant: bool,
    /// Fail if Qdrant is not available (production mode)
    pub require_qdrant: bool,
    /// Custom Qdrant configuration
    pub qdrant_config: Option<QdrantConfig>,
}

impl Default for VectorStorageConfig {
    fn default() -> Self {
        Self {
            prefer_qdrant: true,
            require_qdrant: false,
            qdrant_config: None,
        }
    }
}

impl VectorStorageConfig {
    /// Create configuration for development environment
    pub fn development() -> Self {
        Self {
            prefer_qdrant: false,  // Use in-memory for faster dev iteration
            require_qdrant: false,
            qdrant_config: None,
        }
    }
    
    /// Create configuration for production environment
    pub fn production() -> Self {
        Self {
            prefer_qdrant: true,
            require_qdrant: true,  // Fail if Qdrant not available
            qdrant_config: Some(QdrantConfig::production()),
        }
    }
}

/// Advanced factory with custom configuration
impl VectorStorageFactory {
    /// Create storage with custom configuration
    pub async fn create_with_config(config: VectorStorageConfig) -> Result<VectorBackend, VectorError> {
        if config.require_qdrant {
            info!("🎯 Production mode: requiring Qdrant backend");
            Self::create_qdrant().await
        } else if config.prefer_qdrant {
            info!("🔄 Trying Qdrant with fallback to in-memory");
            Self::create().await
        } else {
            info!("💾 Development mode: using in-memory backend");
            Ok(Self::create_in_memory_backend())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_storage_config_default() {
        let config = VectorStorageConfig::default();
        assert!(config.prefer_qdrant);
        assert!(!config.require_qdrant);
        assert!(config.qdrant_config.is_none());
    }

    #[test]
    fn test_vector_storage_config_development() {
        let config = VectorStorageConfig::development();
        assert!(!config.prefer_qdrant);
        assert!(!config.require_qdrant);
    }

    #[test]
    fn test_vector_storage_config_production() {
        let config = VectorStorageConfig::production();
        assert!(config.prefer_qdrant);
        assert!(config.require_qdrant);
        assert!(config.qdrant_config.is_some());
    }

    #[tokio::test]
    async fn test_in_memory_backend_creation() {
        let backend = VectorStorageFactory::create_in_memory_backend();
        assert_eq!(backend.backend_type(), "in-memory");
        assert!(!backend.is_production_ready());
    }

    #[tokio::test]
    async fn test_backend_stats() {
        let backend = VectorStorageFactory::create_in_memory_backend();
        let stats = backend.stats().await;
        assert_eq!(stats.total_vectors, 0);
    }
}