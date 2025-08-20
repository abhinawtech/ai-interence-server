// PERFORMANCE-OPTIMIZED EMBEDDING SERVICE
// Target: 13.4 TPS → 17+ TPS (25%+ improvement)

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use uuid::Uuid;

use crate::vector::{
    EmbeddingService, EmbeddingResult, EmbeddingMethod,
    EmbeddingCache, CommonQueryCache,
    VectorPoint, VectorError, VectorResult
};
use super::fast_search::{FastVectorSearch, SearchResult};

/// High-performance embedding service with caching and optimized search
pub struct OptimizedEmbeddingService {
    base_service: Arc<RwLock<EmbeddingService>>,
    embedding_cache: Arc<EmbeddingCache>,
    fast_search: Arc<FastVectorSearch>,
    common_cache: Arc<CommonQueryCache>,
    dimension: usize,
}

impl OptimizedEmbeddingService {
    pub async fn new(base_service: EmbeddingService, dimension: usize) -> Self {
        let embedding_cache = Arc::new(EmbeddingCache::new(10000, 24)); // 10k embeddings, 24h TTL
        let common_cache = Arc::new(CommonQueryCache::new());
        let fast_search = Arc::new(FastVectorSearch::new(dimension));
        
        let optimized_service = Self {
            base_service: Arc::new(RwLock::new(base_service)),
            embedding_cache: Arc::clone(&embedding_cache),
            fast_search,
            common_cache: Arc::clone(&common_cache),
            dimension,
        };
        
        // Warm cache with common queries (background task)
        let cache_clone = Arc::clone(&embedding_cache);
        let common_clone = Arc::clone(&common_cache);
        let service_clone = Arc::clone(&optimized_service.base_service);
        
        tokio::spawn(async move {
            let service = service_clone.read().await;
            common_clone.warm_cache(&cache_clone, &*service).await;
        });
        
        optimized_service
    }

    /// Create embedding with aggressive caching (4000ms → <10ms for cached)
    pub async fn create_embedding_fast(&self, text: &str) -> EmbeddingResult {
        self.embedding_cache.get_or_compute(text, |text| async move {
            let service = self.base_service.read().await;
            service.create_embedding(&text).await
        }).await
    }

    /// Optimized semantic search (4000ms → <50ms)
    pub async fn semantic_search(
        &self,
        query: &str,
        limit: usize,
        min_relevance: f32,
    ) -> VectorResult<Vec<SemanticSearchResult>> {
        let search_start = std::time::Instant::now();
        
        // Step 1: Get query embedding (fast via cache)
        let embedding_result = self.create_embedding_fast(query).await
            .map_err(|e| VectorError::Operation(format!("Embedding failed: {}", e)))?;
        
        let embedding_time = search_start.elapsed().as_millis();
        tracing::info!("🔍 Query embedding created in {}ms", embedding_time);
        
        // Step 2: Fast vector search 
        let search_results = self.fast_search.search_similar(
            &embedding_result.embedding,
            limit,
            min_relevance,
        ).await?;
        
        let total_time = search_start.elapsed().as_millis();
        tracing::info!("⚡ Semantic search completed: {} results in {}ms", 
                      search_results.len(), total_time);
        
        // Convert to semantic search results
        let semantic_results = search_results.into_iter().map(|result| {
            SemanticSearchResult {
                id: result.id,
                content: result.metadata.get("conversation")
                    .unwrap_or(&"".to_string()).clone(),
                relevance_score: result.similarity,
                metadata: result.metadata,
                session_id: result.metadata.get("session_id")
                    .unwrap_or(&"unknown".to_string()).clone(),
            }
        }).collect();
        
        Ok(semantic_results)
    }

    /// Store conversation with optimized indexing
    pub async fn store_conversation(
        &self,
        conversation: &str,
        session_id: &str,
    ) -> VectorResult<Uuid> {
        // Create embedding
        let embedding_result = self.create_embedding_fast(conversation).await
            .map_err(|e| VectorError::Operation(format!("Embedding failed: {}", e)))?;
        
        // Create vector point
        let mut metadata = HashMap::new();
        metadata.insert("conversation".to_string(), conversation.to_string());
        metadata.insert("session_id".to_string(), session_id.to_string());
        metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        metadata.insert("type".to_string(), "conversation".to_string());
        
        let vector_point = VectorPoint {
            id: Uuid::new_v4(),
            vector: embedding_result.embedding,
            metadata,
        };
        
        // Store in fast search index
        let id = self.fast_search.insert(vector_point).await?;
        
        tracing::info!("💾 Conversation stored with fast indexing: {}", id);
        Ok(id)
    }

    /// Get performance metrics
    pub async fn get_performance_stats(&self) -> OptimizedEmbeddingStats {
        let cache_stats = self.embedding_cache.stats().await;
        let search_stats = self.fast_search.search_stats().await;
        
        OptimizedEmbeddingStats {
            cache_hit_rate: cache_stats.hit_rate_estimate,
            cache_entries: cache_stats.total_entries,
            cache_memory_mb: cache_stats.memory_usage_mb,
            search_memory_mb: search_stats.memory_usage_mb,
            expected_search_time_ms: search_stats.expected_search_time_ms,
            total_vectors: search_stats.total_vectors,
            embedding_dimension: self.dimension,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticSearchResult {
    pub id: Uuid,
    pub content: String,
    pub relevance_score: f32,
    pub metadata: HashMap<String, String>,
    pub session_id: String,
}

#[derive(Debug)]
pub struct OptimizedEmbeddingStats {
    pub cache_hit_rate: f64,
    pub cache_entries: usize,
    pub cache_memory_mb: usize,
    pub search_memory_mb: usize,
    pub expected_search_time_ms: u64,
    pub total_vectors: usize,
    pub embedding_dimension: usize,
}

// Factory function to create optimized service
pub async fn create_optimized_embedding_service(
    dimension: usize,
) -> Result<OptimizedEmbeddingService, String> {
    // Create base embedding service
    use crate::vector::{EmbeddingConfig, EmbeddingMethod};
    let config = EmbeddingConfig {
        method: EmbeddingMethod::Simple, // Use simple method for now
        dimensions: dimension,
        model_name: Some("optimized-embeddings".to_string()),
        batch_size: 32,
        cache_size: 10000,
    };
    
    let base_service = EmbeddingService::new(config)
        .map_err(|e| format!("Failed to create base embedding service: {}", e))?;
    
    Ok(OptimizedEmbeddingService::new(base_service, dimension).await)
}