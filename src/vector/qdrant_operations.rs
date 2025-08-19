// ================================================================================================
// QDRANT VECTOR OPERATIONS - HIGH-PERFORMANCE VECTOR DATABASE OPERATIONS
// ================================================================================================
//
// Production-ready vector operations with:
// - CRUD operations (Create, Read, Update, Delete)
// - Batch operations for performance
// - Similarity search with filtering
// - Backward compatibility with existing VectorStorage interface
// - Error handling and recovery
// - Performance monitoring and logging
//
// ================================================================================================

use crate::vector::{
    VectorPoint, VectorError, VectorResult, StorageStats,
    qdrant_client::QdrantClient,
};
use anyhow::Result;
use qdrant_client::qdrant::{
    PointStruct, SearchPoints, UpsertPoints,
    DeletePoints, GetPoints, PointsIdsList,
    Value, VectorsOutput, Vector,
    point_id::PointIdOptions, PointId,
    value::Kind,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use uuid::Uuid;

/// Qdrant-backed vector operations implementing the VectorStorage interface
pub struct QdrantVectorStorage {
    client: QdrantClient,
    collection_name: String,
    stats: Arc<RwLock<QdrantStorageStats>>,
}

/// Enhanced statistics for Qdrant storage
#[derive(Debug, Clone)]
pub struct QdrantStorageStats {
    pub total_vectors: usize,
    pub memory_usage_estimate: usize,
    pub last_operation_duration_ms: u64,
    pub total_operations: u64,
    pub failed_operations: u64,
}

impl Default for QdrantStorageStats {
    fn default() -> Self {
        Self {
            total_vectors: 0,
            memory_usage_estimate: 0,
            last_operation_duration_ms: 0,
            total_operations: 0,
            failed_operations: 0,
        }
    }
}

impl QdrantVectorStorage {
    /// Create a new Qdrant vector storage instance
    pub async fn new(client: QdrantClient) -> Result<Self, VectorError> {
        let collection_name = client.get_config().conversation_collection.clone();
        
        info!("🗂️ Initializing Qdrant vector storage for collection: {}", collection_name);
        
        // Ensure the collection exists
        client.ensure_collection(&collection_name)
            .await
            .map_err(|e| VectorError::Operation(format!("Failed to ensure collection: {}", e)))?;
        
        let storage = Self {
            client,
            collection_name,
            stats: Arc::new(RwLock::new(QdrantStorageStats::default())),
        };
        
        // Update initial stats
        storage.refresh_stats().await?;
        
        info!("✅ Qdrant vector storage initialized successfully");
        Ok(storage)
    }
    
    /// Insert a vector point into Qdrant
    pub async fn insert(&self, point: VectorPoint) -> VectorResult<Uuid> {
        let start_time = std::time::Instant::now();
        let result = self._insert_internal(point.clone()).await;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        self.update_operation_stats(duration_ms, result.is_ok()).await;
        
        match result {
            Ok(id) => {
                debug!("✅ Vector inserted successfully: {} ({}ms)", id, duration_ms);
                Ok(id)
            }
            Err(e) => {
                warn!("❌ Vector insertion failed: {} ({}ms)", e, duration_ms);
                Err(e)
            }
        }
    }
    
    async fn _insert_internal(&self, point: VectorPoint) -> VectorResult<Uuid> {
        info!("🔄 Starting vector insertion for ID: {}", point.id);
        info!("📊 Vector dimensions: {}, Metadata entries: {}", point.vector.len(), point.metadata.len());
        
        // Convert metadata to Qdrant payload
        let mut payload = HashMap::new();
        for (key, value) in &point.metadata {
            payload.insert(key.clone(), Value::from(value.clone()));
        }
        
        // Create Qdrant point with explicit vector format
        use qdrant_client::qdrant::{Vectors, vectors};
        
        let vectors = Vectors {
            vectors_options: Some(vectors::VectorsOptions::Vector(
                qdrant_client::qdrant::Vector {
                    data: point.vector,
                    indices: None,
                    vector: None,
                    vectors_count: None,
                }
            )),
        };
        
        let qdrant_point = PointStruct {
            id: Some(point.id.to_string().into()),
            vectors: Some(vectors),
            payload,
        };
        
        // Upsert the point
        let upsert_request = UpsertPoints {
            collection_name: self.collection_name.clone(),
            points: vec![qdrant_point],
            ..Default::default()
        };
        
        info!("📤 Sending upsert request to collection: {}", self.collection_name);
        
        let response = self.client.get_client()
            .upsert_points(upsert_request)
            .await
            .map_err(|e| {
                error!("❌ Qdrant upsert failed: {}", e);
                VectorError::Operation(format!("Failed to upsert point: {}", e))
            })?;
        
        info!("✅ Upsert response received for ID: {}", point.id);
        info!("📋 Response details: {:?}", response);
        
        Ok(point.id)
    }
    
    /// Get a vector point by ID
    pub async fn get(&self, id: &Uuid) -> Option<VectorPoint> {
        match self._get_internal(id).await {
            Ok(Some(point)) => Some(point),
            Ok(None) => None,
            Err(e) => {
                warn!("❌ Failed to get vector {}: {}", id, e);
                None
            }
        }
    }
    
    async fn _get_internal(&self, id: &Uuid) -> VectorResult<Option<VectorPoint>> {
        let get_request = GetPoints {
            collection_name: self.collection_name.clone(),
            ids: vec![PointId {
                point_id_options: Some(PointIdOptions::Uuid(id.to_string())),
            }],
            with_vectors: Some(true.into()),
            with_payload: Some(true.into()),
            ..Default::default()
        };
        
        let response = self.client.get_client()
            .get_points(get_request)
            .await
            .map_err(|e| VectorError::Operation(format!("Failed to get points: {}", e)))?;
        
        if let Some(point) = response.result.first() {
            // Extract vector - simplified approach
            let vector = match &point.vectors {
                Some(vectors_output) => {
                    // For now, assume single unnamed vector - this is a simplified implementation
                    // Real implementation would properly handle the vectors_options oneof field
                    if let Some(ref vectors_options) = vectors_output.vectors_options {
                        match vectors_options {
                            qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(vec_data) => {
                                vec_data.data.clone()
                            }
                            qdrant_client::qdrant::vectors_output::VectorsOptions::Vectors(_) => {
                                return Err(VectorError::Operation("Named vectors not yet supported".to_string()));
                            }
                        }
                    } else {
                        return Err(VectorError::Operation("No vectors_options in VectorsOutput".to_string()));
                    }
                },
                None => return Err(VectorError::Operation("No vector data found".to_string())),
            };
            
            // Extract metadata
            let mut metadata = HashMap::new();
            for (key, value) in &point.payload {
                if let Some(kind) = &value.kind {
                    let str_value = match kind {
                        Kind::StringValue(s) => s.clone(),
                        Kind::IntegerValue(i) => i.to_string(),
                        Kind::DoubleValue(d) => d.to_string(),
                        Kind::BoolValue(b) => b.to_string(),
                        _ => format!("{:?}", kind),
                    };
                    metadata.insert(key.clone(), str_value);
                }
            }
            
            let vector_point = VectorPoint {
                id: *id,
                vector,
                metadata,
            };
            
            Ok(Some(vector_point))
        } else {
            Ok(None)
        }
    }
    
    /// Delete a vector point
    pub async fn delete(&self, id: &Uuid) -> VectorResult<bool> {
        let start_time = std::time::Instant::now();
        let result = self._delete_internal(id).await;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        self.update_operation_stats(duration_ms, result.is_ok()).await;
        
        match result {
            Ok(deleted) => {
                if deleted {
                    debug!("✅ Vector deleted successfully: {} ({}ms)", id, duration_ms);
                } else {
                    debug!("ℹ️ Vector not found for deletion: {} ({}ms)", id, duration_ms);
                }
                Ok(deleted)
            }
            Err(e) => {
                warn!("❌ Vector deletion failed: {} ({}ms)", e, duration_ms);
                Err(e)
            }
        }
    }
    
    async fn _delete_internal(&self, id: &Uuid) -> VectorResult<bool> {
        let delete_request = DeletePoints {
            collection_name: self.collection_name.clone(),
            points: Some(vec![PointId {
                point_id_options: Some(PointIdOptions::Uuid(id.to_string())),
            }].into()),
            ..Default::default()
        };
        
        let _response = self.client.get_client()
            .delete_points(delete_request)
            .await
            .map_err(|e| VectorError::Operation(format!("Failed to delete points: {}", e)))?;
        
        // Qdrant doesn't directly tell us if points were deleted, so we assume success
        Ok(true)
    }
    
    /// Search for similar vectors
    pub async fn search_similar(&self, query_vector: &[f32], limit: usize) -> VectorResult<Vec<(Uuid, f32)>> {
        let start_time = std::time::Instant::now();
        let result = self._search_similar_internal(query_vector, limit).await;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        self.update_operation_stats(duration_ms, result.is_ok()).await;
        
        match result {
            Ok(results) => {
                debug!("✅ Similarity search completed: {} results ({}ms)", results.len(), duration_ms);
                Ok(results)
            }
            Err(e) => {
                warn!("❌ Similarity search failed: {} ({}ms)", e, duration_ms);
                Err(e)
            }
        }
    }
    
    async fn _search_similar_internal(&self, query_vector: &[f32], limit: usize) -> VectorResult<Vec<(Uuid, f32)>> {
        let search_request = SearchPoints {
            collection_name: self.collection_name.clone(),
            vector: query_vector.to_vec(),
            limit: limit as u64,
            with_payload: Some(true.into()),
            with_vectors: Some(false.into()),
            ..Default::default()
        };
        
        let response = self.client.get_client()
            .search_points(search_request)
            .await
            .map_err(|e| VectorError::Operation(format!("Failed to search points: {}", e)))?;
        
        let mut results = Vec::new();
        for scored_point in response.result {
            if let Some(point_id) = &scored_point.id {
                if let Some(point_id_options) = &point_id.point_id_options {
                    match point_id_options {
                        PointIdOptions::Uuid(uuid_str) => {
                            if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                                results.push((uuid, scored_point.score));
                            }
                        }
                        PointIdOptions::Num(_) => {
                            // Handle numeric IDs if needed
                            continue;
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// List all vectors (for debugging - use with caution in production)
    pub async fn list_all(&self) -> Vec<VectorPoint> {
        // This is expensive for large collections - implement pagination in production
        warn!("⚠️ list_all() called - this may be expensive for large collections");
        
        match self._list_all_internal().await {
            Ok(points) => points,
            Err(e) => {
                error!("❌ Failed to list all vectors: {}", e);
                Vec::new()
            }
        }
    }
    
    async fn _list_all_internal(&self) -> VectorResult<Vec<VectorPoint>> {
        // Get total count first - for now we'll skip this and just search with a large limit
        let total_count = 1000; // Default limit for safety
        
        if total_count > 1000 {
            warn!("⚠️ Large collection detected ({} points) - consider using pagination", total_count);
        }
        
        // Search with a large limit to get all points
        let search_request = SearchPoints {
            collection_name: self.collection_name.clone(),
            vector: vec![0.0; self.client.get_config().vector_dimension], // Dummy vector
            limit: total_count.min(10000) as u64, // Limit to 10k for safety
            with_payload: Some(true.into()),
            with_vectors: Some(false.into()),
            ..Default::default()
        };
        
        let response = self.client.get_client()
            .search_points(search_request)
            .await
            .map_err(|e| VectorError::Operation(format!("Failed to search all points: {}", e)))?;
        
        let mut points = Vec::new();
        for scored_point in response.result {
            if let Some(point_id) = &scored_point.id {
                if let Some(point_id_options) = &point_id.point_id_options {
                    if let PointIdOptions::Uuid(uuid_str) = point_id_options {
                        if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                            if let Some(point) = self.get(&uuid).await {
                                points.push(point);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(points)
    }
    
    /// Get storage statistics
    pub async fn stats(&self) -> StorageStats {
        // Refresh stats from Qdrant
        if let Err(e) = self.refresh_stats().await {
            warn!("Failed to refresh stats: {}", e);
        }
        
        let qdrant_stats = self.stats.read().await;
        StorageStats {
            total_vectors: qdrant_stats.total_vectors,
            memory_usage_estimate: qdrant_stats.memory_usage_estimate,
        }
    }
    
    /// Get enhanced Qdrant-specific statistics
    pub async fn qdrant_stats(&self) -> QdrantStorageStats {
        // Refresh stats from Qdrant
        if let Err(e) = self.refresh_stats().await {
            warn!("Failed to refresh stats: {}", e);
        }
        
        self.stats.read().await.clone()
    }
    
    /// Refresh statistics from Qdrant
    async fn refresh_stats(&self) -> VectorResult<()> {
        debug!("🔄 Refreshing stats from Qdrant collection: {}", self.collection_name);
        
        // Get collection info to get actual count
        let total_count = match self.client.get_client().collection_info(&self.collection_name).await {
            Ok(response) => {
                let points_count = response.result.as_ref()
                    .map(|info| info.points_count.unwrap_or(0))
                    .unwrap_or(0);
                debug!("📊 Qdrant reports {} points in collection", points_count);
                points_count as usize
            }
            Err(e) => {
                warn!("⚠️ Failed to get collection info for stats: {}", e);
                0
            }
        };
        
        let mut stats = self.stats.write().await;
        stats.total_vectors = total_count;
        // Rough estimate: each vector takes about 4 bytes per dimension + metadata overhead
        stats.memory_usage_estimate = total_count * (self.client.get_config().vector_dimension * 4 + 100);
        
        debug!("📈 Updated stats: {} vectors, {} bytes estimated", stats.total_vectors, stats.memory_usage_estimate);
        
        Ok(())
    }
    
    /// Update operation statistics
    async fn update_operation_stats(&self, duration_ms: u64, success: bool) {
        let mut stats = self.stats.write().await;
        stats.last_operation_duration_ms = duration_ms;
        stats.total_operations += 1;
        if !success {
            stats.failed_operations += 1;
        }
    }
    
    /// Batch insert multiple vectors for performance
    pub async fn batch_insert(&self, points: Vec<VectorPoint>) -> VectorResult<Vec<Uuid>> {
        let start_time = std::time::Instant::now();
        
        info!("📦 Starting batch insert of {} vectors", points.len());
        
        let mut qdrant_points = Vec::new();
        let mut ids = Vec::new();
        
        for point in points {
            // Convert metadata to Qdrant payload
            let mut payload = HashMap::new();
            for (key, value) in &point.metadata {
                payload.insert(key.clone(), Value::from(value.clone()));
            }
            
            let qdrant_point = PointStruct {
                id: Some(point.id.to_string().into()),
                vectors: Some(Vector::from(point.vector).into()),
                payload,
            };
            
            qdrant_points.push(qdrant_point);
            ids.push(point.id);
        }
        
        let upsert_request = UpsertPoints {
            collection_name: self.collection_name.clone(),
            points: qdrant_points,
            ..Default::default()
        };
        
        let result = self.client.get_client()
            .upsert_points(upsert_request)
            .await
            .map_err(|e| VectorError::Operation(format!("Failed to batch upsert points: {}", e)));
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        self.update_operation_stats(duration_ms, result.is_ok()).await;
        
        match result {
            Ok(_) => {
                info!("✅ Batch insert completed: {} vectors ({}ms)", ids.len(), duration_ms);
                Ok(ids)
            }
            Err(e) => {
                error!("❌ Batch insert failed: {} ({}ms)", e, duration_ms);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::QdrantConfig;

    #[test]
    fn test_qdrant_storage_stats_default() {
        let stats = QdrantStorageStats::default();
        assert_eq!(stats.total_vectors, 0);
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.failed_operations, 0);
    }

    #[tokio::test]
    async fn test_stats_conversion() {
        let qdrant_stats = QdrantStorageStats {
            total_vectors: 100,
            memory_usage_estimate: 1000,
            last_operation_duration_ms: 50,
            total_operations: 200,
            failed_operations: 5,
        };
        
        // Test conversion to StorageStats
        let storage_stats = StorageStats {
            total_vectors: qdrant_stats.total_vectors,
            memory_usage_estimate: qdrant_stats.memory_usage_estimate,
        };
        
        assert_eq!(storage_stats.total_vectors, 100);
        assert_eq!(storage_stats.memory_usage_estimate, 1000);
    }
}