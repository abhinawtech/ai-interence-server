// ================================================================================================
// VECTOR DATABASE MODULE - MINIMAL IMPLEMENTATION
// ================================================================================================
//
// This is a clean, step-by-step implementation of vector database functionality.
// Each component is implemented and tested incrementally.
//
// ================================================================================================

pub mod storage;
pub mod embedding;
pub mod embedding_service;
pub mod qdrant_config;
pub mod qdrant_client;
pub mod qdrant_operations;
pub mod storage_factory;
pub mod index_optimizer;
pub mod reindex_manager;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// Re-export key types
pub use storage::{VectorStorage, StorageStats};
pub use embedding::create_simple_embedding;
pub use embedding_service::{EmbeddingService, EmbeddingConfig, EmbeddingResult, EmbeddingMethod, EmbeddingServiceStats};
pub use qdrant_config::QdrantConfig;
pub use qdrant_client::{QdrantClient, QdrantClientBuilder};
pub use qdrant_operations::{QdrantVectorStorage, QdrantStorageStats};
pub use storage_factory::{VectorBackend, VectorStorageFactory, VectorStorageConfig};
pub use index_optimizer::{IndexOptimizer, IndexProfile, IndexPerformanceMetrics, CollectionIndexConfig, create_index_optimizer};
pub use reindex_manager::{ReindexManager, ReindexJobStatus, ReindexJobState, ReindexJobConfig, JobPriority, ResourceUsage, create_reindex_manager};

// ================================================================================================
// BASIC TYPES
// ================================================================================================

/// A vector point for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPoint {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, String>, // Keep simple for now
}

impl VectorPoint {
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            id: Uuid::new_v4(),
            vector,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(vector: Vec<f32>, metadata: HashMap<String, String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            vector,
            metadata,
        }
    }
}

/// Simple vector error type
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("Vector operation failed: {0}")]
    Operation(String),
}

pub type VectorResult<T> = Result<T, VectorError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_point_creation() {
        let vector = vec![1.0, 2.0, 3.0];
        let point = VectorPoint::new(vector.clone());
        assert_eq!(point.vector, vector);
        assert!(point.metadata.is_empty());
    }

    #[test]
    fn test_vector_point_with_metadata() {
        let vector = vec![1.0, 2.0, 3.0];
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());
        
        let point = VectorPoint::with_metadata(vector.clone(), metadata.clone());
        assert_eq!(point.vector, vector);
        assert_eq!(point.metadata, metadata);
    }
}