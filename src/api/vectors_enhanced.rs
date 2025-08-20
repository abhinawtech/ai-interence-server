// ================================================================================================
// ENHANCED VECTOR API - SEMANTIC SEARCH AND METADATA FILTERING
// ================================================================================================
//
// Production-grade vector operations with semantic intelligence:
// - Text-based semantic search (no manual vectors needed)
// - Advanced metadata filtering and querying
// - Hybrid search combining semantic + keyword filtering
// - Performance optimization with caching
// - Integration with embedding service for automatic vectorization
//
// This transforms the basic vector API into a semantic search engine.
//
// ================================================================================================

use crate::vector::{VectorPoint, VectorBackend, EmbeddingService};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};
use uuid::Uuid;

// ================================================================================================
// ENHANCED API TYPES
// ================================================================================================

/// Semantic search request using natural text
#[derive(Debug, Deserialize)]
pub struct SemanticSearchRequest {
    pub query: String,                                      // Natural language query
    pub limit: Option<usize>,                              // Number of results
    pub metadata_filters: Option<HashMap<String, String>>, // Filter by metadata
    pub similarity_threshold: Option<f32>,                 // Minimum similarity score
    pub include_vectors: Option<bool>,                     // Return vectors in response
}

/// Advanced vector search with filtering
#[derive(Debug, Deserialize)]
pub struct AdvancedSearchRequest {
    pub vector: Option<Vec<f32>>,                          // Optional vector (can use text instead)
    pub text: Option<String>,                              // Alternative to vector
    pub limit: Option<usize>,
    pub metadata_filters: Option<HashMap<String, String>>,
    pub similarity_threshold: Option<f32>,
    pub boost_recent: Option<bool>,                        // Boost recent entries
    pub category_filter: Option<Vec<String>>,              // Filter by categories
}

/// Enhanced search result with more information
#[derive(Debug, Serialize)]
pub struct EnhancedSearchResult {
    pub id: Uuid,
    pub similarity: f32,
    pub metadata: HashMap<String, String>,
    pub vector: Option<Vec<f32>>,                          // Optional vector data
    pub text_excerpt: Option<String>,                     // Text excerpt if available
    pub relevance_score: f32,                             // Combined relevance score
    pub category: Option<String>,                         // Auto-detected category
}

/// Enhanced search response
#[derive(Debug, Serialize)]
pub struct EnhancedSearchResponse {
    pub results: Vec<EnhancedSearchResult>,
    pub total_found: usize,
    pub query_processed: String,                          // Processed/normalized query
    pub search_time_ms: u64,
    pub embedding_time_ms: Option<u64>,                   // Time for embedding generation
    pub filters_applied: Vec<String>,                     // List of applied filters
}

/// Text-based vector insertion
#[derive(Debug, Deserialize)]
pub struct TextVectorRequest {
    pub text: String,
    pub metadata: Option<HashMap<String, String>>,
    pub category: Option<String>,
    pub auto_categorize: Option<bool>,                    // Auto-detect category
}

/// Enhanced insertion response
#[derive(Debug, Serialize)]
pub struct TextVectorResponse {
    pub id: Uuid,
    pub success: bool,
    pub message: String,
    pub embedding_method: String,                         // How vector was generated
    pub auto_category: Option<String>,                   // Auto-detected category
    pub processing_time_ms: u64,
}

/// Batch operations for multiple texts
#[derive(Debug, Deserialize)]
pub struct BatchTextVectorRequest {
    pub items: Vec<TextVectorRequest>,
    pub deduplicate: Option<bool>,                       // Remove similar items
    pub similarity_threshold: Option<f32>,              // For deduplication
}

/// Query expansion and analysis
#[derive(Debug, Serialize)]
pub struct QueryAnalysis {
    pub original_query: String,
    pub expanded_queries: Vec<String>,                   // Related queries
    pub detected_intent: Option<String>,                // Query intent
    pub key_concepts: Vec<String>,                      // Extracted concepts
    pub suggested_filters: HashMap<String, String>,     // Suggested metadata filters
}

// ================================================================================================
// API STATE
// ================================================================================================

pub type EnhancedVectorApiState = (Arc<VectorBackend>, Arc<RwLock<EmbeddingService>>);

// ================================================================================================
// ENHANCED API HANDLERS
// ================================================================================================

/// Semantic search using natural language
pub async fn semantic_search(
    State((vector_backend, embedding_service)): State<EnhancedVectorApiState>,
    Json(request): Json<SemanticSearchRequest>,
) -> Result<Json<EnhancedSearchResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    info!("🔍 Semantic search query: '{}'", request.query);
    
    // Generate embedding for the query
    let embedding_start = std::time::Instant::now();
    let embedding_service = embedding_service.read().await;
    let embedding_result = embedding_service.embed_text(&request.query)
        .await
        .map_err(|e| {
            warn!("Failed to generate embedding: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    drop(embedding_service);
    
    let embedding_time = embedding_start.elapsed().as_millis() as u64;
    
    // Search for similar vectors
    let limit = request.limit.unwrap_or(10);
    let similar_results = vector_backend.search_similar(&embedding_result.vector, limit * 2)
        .await
        .unwrap_or_default();
    
    // Apply filters and enhance results
    let mut enhanced_results = Vec::new();
    let mut filters_applied = Vec::new();
    
    for (id, similarity) in similar_results {
        // Apply similarity threshold filter
        if let Some(threshold) = request.similarity_threshold {
            if similarity < threshold {
                continue;
            }
        }
        
        if let Some(point) = vector_backend.get(&id).await {
            // Apply metadata filters
            if let Some(ref filters) = request.metadata_filters {
                let mut matches_filters = true;
                for (key, value) in filters {
                    if !point.metadata.get(key).map_or(false, |v| v.contains(value)) {
                        matches_filters = false;
                        break;
                    }
                }
                if !matches_filters {
                    continue;
                }
                if filters_applied.is_empty() {
                    filters_applied.push("metadata".to_string());
                }
            }
            
            // Create enhanced result
            let text_excerpt = point.metadata.get("source_text")
                .or_else(|| point.metadata.get("text"))
                .map(|text| {
                    if text.len() > 100 {
                        format!("{}...", &text[..97])
                    } else {
                        text.clone()
                    }
                });
            
            let category = point.metadata.get("category")
                .or_else(|| point.metadata.get("type"))
                .cloned();
            
            let relevance_score = calculate_relevance_score(similarity, &point, &request);
            
            enhanced_results.push(EnhancedSearchResult {
                id,
                similarity,
                metadata: point.metadata,
                vector: if request.include_vectors.unwrap_or(false) {
                    Some(point.vector)
                } else {
                    None
                },
                text_excerpt,
                relevance_score,
                category,
            });
            
            if enhanced_results.len() >= limit {
                break;
            }
        }
    }
    
    // Sort by relevance score
    enhanced_results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
    
    let search_time = start_time.elapsed().as_millis() as u64;
    
    info!("✅ Semantic search completed: {} results in {}ms", enhanced_results.len(), search_time);
    
    Ok(Json(EnhancedSearchResponse {
        total_found: enhanced_results.len(),
        results: enhanced_results,
        query_processed: request.query,
        search_time_ms: search_time,
        embedding_time_ms: Some(embedding_time),
        filters_applied,
    }))
}

/// Insert text and automatically generate vector
pub async fn insert_text_vector(
    State((vector_backend, embedding_service)): State<EnhancedVectorApiState>,
    Json(request): Json<TextVectorRequest>,
) -> Result<Json<TextVectorResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    info!("📝 Inserting text vector: '{}'", 
          if request.text.len() > 50 { &request.text[..50] } else { &request.text });
    
    // Generate embedding
    let embedding_service = embedding_service.read().await;
    let mut enhanced_metadata = request.metadata.unwrap_or_default();
    
    // Auto-categorization
    let auto_category = if request.auto_categorize.unwrap_or(false) {
        Some(detect_category(&request.text))
    } else {
        None
    };
    
    if let Some(ref category) = request.category.or(auto_category.clone()) {
        enhanced_metadata.insert("category".to_string(), category.clone());
    }
    
    // Create vector point
    let vector_point = embedding_service.create_vector_point(&request.text, enhanced_metadata)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let embedding_method = format!("{:?}", 
        embedding_service.embed_text(&request.text).await
            .map(|r| r.method)
            .unwrap_or(crate::vector::EmbeddingMethod::TfIdf)
    );
    
    drop(embedding_service);
    
    let id = vector_point.id;
    
    // Insert into vector storage
    vector_backend.insert(vector_point)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let processing_time = start_time.elapsed().as_millis() as u64;
    
    Ok(Json(TextVectorResponse {
        id,
        success: true,
        message: "Text vectorized and stored successfully".to_string(),
        embedding_method,
        auto_category,
        processing_time_ms: processing_time,
    }))
}

/// Advanced search with multiple options
pub async fn advanced_search(
    State((vector_backend, embedding_service)): State<EnhancedVectorApiState>,
    Json(request): Json<AdvancedSearchRequest>,
) -> Result<Json<EnhancedSearchResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    // Determine search vector
    let (query_vector, embedding_time, query_text) = if let Some(text) = &request.text {
        let embedding_start = std::time::Instant::now();
        let embedding_service = embedding_service.read().await;
        let result = embedding_service.embed_text(text)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        drop(embedding_service);
        
        let time = embedding_start.elapsed().as_millis() as u64;
        (result.vector, Some(time), text.clone())
    } else if let Some(vector) = &request.vector {
        (vector.clone(), None, "vector-based".to_string())
    } else {
        return Err(StatusCode::BAD_REQUEST);
    };
    
    // Search for similar vectors
    let limit = request.limit.unwrap_or(10);
    let similar_results = vector_backend.search_similar(&query_vector, limit * 3)
        .await
        .unwrap_or_default();
    
    // Apply advanced filtering
    let mut enhanced_results = Vec::new();
    let mut filters_applied = Vec::new();
    
    for (id, similarity) in similar_results {
        if let Some(point) = vector_backend.get(&id).await {
            // Similarity threshold filter
            if let Some(threshold) = request.similarity_threshold {
                if similarity < threshold {
                    continue;
                }
            }
            
            // Metadata filters
            if let Some(ref filters) = request.metadata_filters {
                let mut matches = true;
                for (key, value) in filters {
                    if !point.metadata.get(key).map_or(false, |v| v.contains(value)) {
                        matches = false;
                        break;
                    }
                }
                if !matches {
                    continue;
                }
                if filters_applied.iter().find(|&x| x == "metadata").is_none() {
                    filters_applied.push("metadata".to_string());
                }
            }
            
            // Category filters
            if let Some(ref categories) = request.category_filter {
                if let Some(point_category) = point.metadata.get("category") {
                    if !categories.contains(point_category) {
                        continue;
                    }
                } else {
                    continue;
                }
                if filters_applied.iter().find(|&x| x == "category").is_none() {
                    filters_applied.push("category".to_string());
                }
            }
            
            let relevance_score = if request.boost_recent.unwrap_or(false) {
                calculate_boosted_relevance_score(similarity, &point)
            } else {
                similarity
            };
            
            enhanced_results.push(EnhancedSearchResult {
                id,
                similarity,
                metadata: point.metadata.clone(),
                vector: None,
                text_excerpt: point.metadata.get("source_text").map(|text| {
                    if text.len() > 100 {
                        format!("{}...", &text[..97])
                    } else {
                        text.clone()
                    }
                }),
                relevance_score,
                category: point.metadata.get("category").cloned(),
            });
        }
    }
    
    // Sort by relevance score
    enhanced_results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
    enhanced_results.truncate(limit);
    
    let search_time = start_time.elapsed().as_millis() as u64;
    
    Ok(Json(EnhancedSearchResponse {
        total_found: enhanced_results.len(),
        results: enhanced_results,
        query_processed: query_text,
        search_time_ms: search_time,
        embedding_time_ms: embedding_time,
        filters_applied,
    }))
}

/// Analyze query and provide suggestions
pub async fn analyze_query(
    State((_vector_backend, embedding_service)): State<EnhancedVectorApiState>,
    Json(query): Json<serde_json::Value>,
) -> Result<Json<QueryAnalysis>, StatusCode> {
    let query_text = query.get("query")
        .and_then(|v| v.as_str())
        .ok_or(StatusCode::BAD_REQUEST)?;
    
    info!("🔍 Analyzing query: '{}'", query_text);
    
    // Simple query analysis (can be enhanced with NLP)
    let key_concepts = extract_key_concepts(query_text);
    let detected_intent = detect_intent(query_text);
    let expanded_queries = generate_expanded_queries(query_text, &key_concepts);
    let suggested_filters = suggest_metadata_filters(query_text, &key_concepts);
    
    Ok(Json(QueryAnalysis {
        original_query: query_text.to_string(),
        expanded_queries,
        detected_intent,
        key_concepts,
        suggested_filters,
    }))
}

/// Get enhanced statistics
pub async fn get_enhanced_stats(
    State((vector_backend, embedding_service)): State<EnhancedVectorApiState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let vector_stats = vector_backend.stats().await;
    let embedding_service = embedding_service.read().await;
    let embedding_stats = embedding_service.get_stats().await;
    drop(embedding_service);
    
    Ok(Json(serde_json::json!({
        "vector_storage": {
            "total_vectors": vector_stats.total_vectors,
            "memory_usage_estimate": vector_stats.memory_usage_estimate
        },
        "embedding_service": embedding_stats,
        "capabilities": {
            "semantic_search": true,
            "metadata_filtering": true,
            "text_vectorization": true,
            "auto_categorization": true,
            "query_analysis": true
        }
    })))
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

fn calculate_relevance_score(similarity: f32, point: &VectorPoint, request: &SemanticSearchRequest) -> f32 {
    let mut score = similarity;
    
    // Boost based on metadata quality
    if point.metadata.contains_key("category") {
        score += 0.05;
    }
    
    if point.metadata.contains_key("source_text") {
        score += 0.03;
    }
    
    // Apply filters boost
    if let Some(ref filters) = request.metadata_filters {
        let matching_filters = filters.iter()
            .filter(|(key, value)| {
                point.metadata.get(*key).map_or(false, |v| v.contains(*value))
            })
            .count();
        score += (matching_filters as f32) * 0.02;
    }
    
    score.min(1.0)
}

fn calculate_boosted_relevance_score(similarity: f32, point: &VectorPoint) -> f32 {
    let mut score = similarity;
    
    // Boost recent entries (if timestamp exists)
    if let Some(timestamp_str) = point.metadata.get("timestamp") {
        if let Ok(timestamp) = chrono::DateTime::parse_from_rfc3339(timestamp_str) {
            let age_hours = chrono::Utc::now().signed_duration_since(timestamp.with_timezone(&chrono::Utc)).num_hours();
            if age_hours < 24 {
                score += 0.1; // Boost recent entries
            }
        }
    }
    
    score.min(1.0)
}

fn detect_category(text: &str) -> String {
    let text_lower = text.to_lowercase();
    
    if text_lower.contains("code") || text_lower.contains("programming") || text_lower.contains("function") {
        "programming".to_string()
    } else if text_lower.contains("question") || text_lower.contains("how") || text_lower.contains("what") {
        "question".to_string()
    } else if text_lower.contains("error") || text_lower.contains("bug") || text_lower.contains("issue") {
        "troubleshooting".to_string()
    } else if text_lower.contains("api") || text_lower.contains("endpoint") || text_lower.contains("service") {
        "api".to_string()
    } else {
        "general".to_string()
    }
}

fn extract_key_concepts(query: &str) -> Vec<String> {
    // Simple keyword extraction (can be enhanced with NLP)
    query.split_whitespace()
        .filter(|word| word.len() > 3)
        .map(|word| word.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .filter(|word| !word.is_empty() && !is_stop_word(word))
        .collect()
}

fn is_stop_word(word: &str) -> bool {
    matches!(word, "the" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by" | "this" | "that" | "these" | "those")
}

fn detect_intent(query: &str) -> Option<String> {
    let query_lower = query.to_lowercase();
    
    if query_lower.starts_with("how") || query_lower.contains("how to") {
        Some("how_to".to_string())
    } else if query_lower.starts_with("what") || query_lower.starts_with("define") {
        Some("definition".to_string())
    } else if query_lower.starts_with("why") || query_lower.contains("because") {
        Some("explanation".to_string())
    } else if query_lower.contains("error") || query_lower.contains("problem") {
        Some("troubleshooting".to_string())
    } else if query_lower.contains("example") || query_lower.contains("show me") {
        Some("example".to_string())
    } else {
        None
    }
}

fn generate_expanded_queries(original: &str, concepts: &[String]) -> Vec<String> {
    let mut expanded = Vec::new();
    
    // Add concept combinations
    if concepts.len() >= 2 {
        for i in 0..concepts.len().min(3) {
            for j in (i+1)..concepts.len().min(3) {
                expanded.push(format!("{} {}", concepts[i], concepts[j]));
            }
        }
    }
    
    // Add synonyms for common terms
    for concept in concepts.iter().take(2) {
        match concept.as_str() {
            "api" => expanded.push("endpoint".to_string()),
            "function" => expanded.push("method".to_string()),
            "error" => expanded.push("bug issue".to_string()),
            "code" => expanded.push("programming".to_string()),
            _ => {}
        }
    }
    
    expanded.truncate(5); // Limit to 5 expanded queries
    expanded
}

fn suggest_metadata_filters(query: &str, concepts: &[String]) -> HashMap<String, String> {
    let mut filters = HashMap::new();
    
    // Suggest category filters based on concepts
    for concept in concepts {
        match concept.as_str() {
            "api" | "endpoint" | "service" => {
                filters.insert("category".to_string(), "api".to_string());
            },
            "code" | "programming" | "function" => {
                filters.insert("category".to_string(), "programming".to_string());
            },
            "error" | "bug" | "issue" => {
                filters.insert("category".to_string(), "troubleshooting".to_string());
            },
            _ => {}
        }
    }
    
    // Suggest type filters
    if query.to_lowercase().contains("question") {
        filters.insert("type".to_string(), "question".to_string());
    }
    
    filters
}

// ================================================================================================
// ROUTER CREATION
// ================================================================================================

pub fn create_enhanced_vector_router() -> Router<EnhancedVectorApiState> {
    Router::new()
        .route("/api/v1/vectors/search/semantic", post(semantic_search))
        .route("/api/v1/vectors/search/advanced", post(advanced_search))
        .route("/api/v1/vectors/text", post(insert_text_vector))
        .route("/api/v1/vectors/analyze", post(analyze_query))
        .route("/api/v1/vectors/stats/enhanced", get(get_enhanced_stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_detection() {
        assert_eq!(detect_category("How to write a function in Python"), "programming");
        assert_eq!(detect_category("API endpoint not working"), "api");
        assert_eq!(detect_category("What is machine learning?"), "question");
        assert_eq!(detect_category("Error in my code"), "troubleshooting");
    }

    #[test]
    fn test_key_concepts_extraction() {
        let concepts = extract_key_concepts("How to implement REST API authentication");
        assert!(concepts.contains(&"implement".to_string()));
        assert!(concepts.contains(&"authentication".to_string()));
        assert!(!concepts.contains(&"how".to_string())); // Stop word
    }

    #[test]
    fn test_intent_detection() {
        assert_eq!(detect_intent("How to solve this problem"), Some("how_to".to_string()));
        assert_eq!(detect_intent("What is React?"), Some("definition".to_string()));
        assert_eq!(detect_intent("API returns error 500"), Some("troubleshooting".to_string()));
    }
}