// ================================================================================================
// SEMANTIC SEARCH API - INTELLIGENT SEARCH WITH SESSION MANAGEMENT
// ================================================================================================
//
// Production-grade semantic search engine with:
// - Session-aware context management for conversational search
// - Multi-domain search across conversations, documents, and knowledge
// - Real-time search suggestions and auto-completion
// - Search personalization and learning from user patterns
// - Cross-reference linking and topic discovery
//
// This creates a Google-like search experience with semantic understanding.
//
// ================================================================================================

use crate::vector::{VectorBackend, EmbeddingService};
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
// SEARCH API TYPES
// ================================================================================================

/// Semantic search request with session context
#[derive(Debug, Deserialize)]
pub struct SemanticSearchRequest {
    pub query: String,                                      // Natural language query
    pub session_id: Option<String>,                        // User session for context
    pub limit: Option<usize>,                              // Number of results (default: 10)
    pub domains: Option<Vec<SearchDomain>>,                // Search domains
    pub filters: Option<SearchFilters>,                    // Advanced filters
    pub personalize: Option<bool>,                         // Use user preferences (default: true)
    pub include_suggestions: Option<bool>,                 // Include search suggestions (default: true)
}

/// Search domains for multi-domain search
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum SearchDomain {
    Conversations,      // Search conversation history
    Documents,          // Search documents and content
    Knowledge,          // Search knowledge base
    Code,              // Search code examples
    All,               // Search everything
}

/// Advanced search filters
#[derive(Debug, Deserialize)]
pub struct SearchFilters {
    pub time_range: Option<TimeRange>,                     // Time-based filtering
    pub content_type: Option<Vec<String>>,                 // Content type filters
    pub categories: Option<Vec<String>>,                   // Category filters
    pub authors: Option<Vec<String>>,                      // Author filters
    pub language: Option<String>,                          // Language preference
    pub quality_threshold: Option<f32>,                    // Minimum quality score
}

/// Time range filter
#[derive(Debug, Deserialize)]
pub struct TimeRange {
    pub start: Option<String>,                             // ISO 8601 timestamp
    pub end: Option<String>,                               // ISO 8601 timestamp
    pub relative: Option<String>,                          // e.g., "1d", "1w", "1m"
}

/// Comprehensive search response
#[derive(Debug, Serialize)]
pub struct SemanticSearchResponse {
    pub results: Vec<SearchResult>,
    pub total_found: usize,
    pub query_processed: String,                           // Processed/expanded query
    pub search_time_ms: u64,
    pub session_context: Option<SessionContext>,           // Session information
    pub suggestions: Option<Vec<SearchSuggestion>>,        // Related suggestions
    pub facets: Option<Vec<SearchFacet>>,                 // Result facets for filtering
    pub personalization: Option<PersonalizationInfo>,     // Personalization details
}

/// Individual search result with enhanced metadata
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: Uuid,
    pub title: String,                                     // Result title
    pub content: String,                                   // Content excerpt
    pub url: Option<String>,                              // Optional URL
    pub score: f32,                                       // Relevance score
    pub domain: SearchDomain,                             // Source domain
    pub metadata: HashMap<String, String>,                // Additional metadata
    pub highlights: Vec<TextHighlight>,                   // Search term highlights
    pub related_results: Vec<String>,                     // Related result IDs
    pub timestamp: Option<String>,                        // Content timestamp
    pub author: Option<String>,                           // Content author
}

/// Text highlighting for search terms
#[derive(Debug, Serialize)]
pub struct TextHighlight {
    pub text: String,                                     // Highlighted text
    pub start: usize,                                     // Start position
    pub end: usize,                                       // End position
    pub relevance: f32,                                   // Highlight relevance
}

/// Search suggestions for query expansion
#[derive(Debug, Serialize)]
pub struct SearchSuggestion {
    pub query: String,                                    // Suggested query
    pub type_: SuggestionType,                           // Suggestion type
    pub confidence: f32,                                 // Confidence score
    pub preview_count: usize,                            // Number of results
}

/// Types of search suggestions
#[derive(Debug, Serialize)]
pub enum SuggestionType {
    Related,            // Related concepts
    Completion,         // Query completion
    Correction,         // Spelling correction
    Alternative,        // Alternative phrasing
    Drill,             // Drill-down suggestion
}

/// Search result facets for filtering
#[derive(Debug, Serialize)]
pub struct SearchFacet {
    pub name: String,                                     // Facet name
    pub values: Vec<FacetValue>,                         // Facet values
}

/// Individual facet value with count
#[derive(Debug, Serialize)]
pub struct FacetValue {
    pub value: String,                                    // Facet value
    pub count: usize,                                    // Result count
    pub selected: bool,                                  // Currently selected
}

/// Session context information
#[derive(Debug, Serialize)]
pub struct SessionContext {
    pub session_id: String,
    pub query_history: Vec<String>,                       // Recent queries
    pub context_topics: Vec<String>,                      // Identified topics
    pub search_intent: Option<SearchIntent>,              // Current intent
    pub personalization_score: f32,                      // Personalization level
}

/// Detected search intent
#[derive(Debug, Serialize)]
pub enum SearchIntent {
    Learning,           // Learning about concepts
    Troubleshooting,    // Solving problems
    Implementation,     // How-to implementation
    Reference,          // Looking up reference
    Exploration,        // Exploratory search
    Comparison,         // Comparing options
}

/// Personalization information
#[derive(Debug, Serialize)]
pub struct PersonalizationInfo {
    pub user_preferences: HashMap<String, f32>,           // User preference scores
    pub search_patterns: Vec<String>,                     // Common search patterns
    pub expertise_level: ExpertiseLevel,                  // Detected expertise
    pub preferred_domains: Vec<SearchDomain>,             // Preferred domains
}

/// User expertise level
#[derive(Debug, Serialize)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Contextual search request (uses session context)
#[derive(Debug, Deserialize)]
pub struct ContextualSearchRequest {
    pub query: String,
    pub session_id: String,                               // Required for context
    pub use_full_context: Option<bool>,                   // Use all session context
    pub context_depth: Option<usize>,                     // Context history depth
}

/// Search trending topics request
#[derive(Debug, Deserialize)]
pub struct TrendingRequest {
    pub domain: Option<SearchDomain>,
    pub time_period: Option<String>,                      // e.g., "24h", "7d", "30d"
    pub limit: Option<usize>,
}

/// Trending topics response
#[derive(Debug, Serialize)]
pub struct TrendingResponse {
    pub topics: Vec<TrendingTopic>,
    pub time_period: String,
    pub total_queries: usize,
    pub generated_at: String,
}

/// Individual trending topic
#[derive(Debug, Serialize)]
pub struct TrendingTopic {
    pub topic: String,
    pub query_count: usize,
    pub growth_rate: f32,                                 // Percentage growth
    pub related_queries: Vec<String>,
    pub top_result_domains: Vec<SearchDomain>,
}

/// Search suggestions request (for auto-complete)
#[derive(Debug, Deserialize)]
pub struct SuggestionsRequest {
    pub partial_query: String,
    pub session_id: Option<String>,
    pub limit: Option<usize>,                             // Default: 5
    pub include_popular: Option<bool>,                    // Include popular searches
}

/// Auto-complete suggestions response
#[derive(Debug, Serialize)]
pub struct SuggestionsResponse {
    pub suggestions: Vec<SearchSuggestion>,
    pub popular_searches: Option<Vec<String>>,            // Popular related searches
    pub personalized: bool,                              // Whether personalized
}

// ================================================================================================
// API STATE AND MANAGEMENT
// ================================================================================================

/// Search session manager for context tracking
#[derive(Debug)]
pub struct SearchSessionManager {
    sessions: Arc<RwLock<HashMap<String, SearchSession>>>,
}

/// Individual search session
#[derive(Debug, Clone)]
pub struct SearchSession {
    pub session_id: String,
    pub query_history: Vec<QueryHistoryEntry>,
    pub context_topics: Vec<String>,
    pub user_preferences: HashMap<String, f32>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub total_queries: usize,
}

/// Query history entry
#[derive(Debug, Clone)]
pub struct QueryHistoryEntry {
    pub query: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub results_count: usize,
    pub clicked_results: Vec<String>,
    pub satisfaction_score: Option<f32>,
}

impl SearchSessionManager {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create session
    pub async fn get_or_create_session(&self, session_id: Option<String>) -> SearchSession {
        let session_id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let mut sessions = self.sessions.write().await;
        
        sessions.entry(session_id.clone()).or_insert_with(|| {
            SearchSession {
                session_id: session_id.clone(),
                query_history: Vec::new(),
                context_topics: Vec::new(),
                user_preferences: HashMap::new(),
                created_at: chrono::Utc::now(),
                last_activity: chrono::Utc::now(),
                total_queries: 0,
            }
        }).clone()
    }

    /// Update session with new query
    pub async fn update_session(&self, session_id: &str, query: &str, results_count: usize) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.query_history.push(QueryHistoryEntry {
                query: query.to_string(),
                timestamp: chrono::Utc::now(),
                results_count,
                clicked_results: Vec::new(),
                satisfaction_score: None,
            });
            session.last_activity = chrono::Utc::now();
            session.total_queries += 1;
            
            // Update context topics (simple keyword extraction)
            let keywords = extract_keywords(query);
            for keyword in keywords {
                if !session.context_topics.contains(&keyword) {
                    session.context_topics.push(keyword);
                }
            }
            
            // Limit history size
            if session.query_history.len() > 50 {
                session.query_history.remove(0);
            }
            if session.context_topics.len() > 20 {
                session.context_topics.remove(0);
            }
        }
    }
}

pub type SearchApiState = (
    Arc<VectorBackend>,
    Arc<RwLock<EmbeddingService>>,
    Arc<SearchSessionManager>,
);

// ================================================================================================
// API HANDLERS
// ================================================================================================

/// Semantic search with session awareness
pub async fn semantic_search(
    State((vector_backend, embedding_service, session_manager)): State<SearchApiState>,
    Json(request): Json<SemanticSearchRequest>,
) -> Result<Json<SemanticSearchResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    info!("🔍 Semantic search: '{}' (session: {:?})", 
          request.query, request.session_id);

    // Get or create session
    let session = session_manager.get_or_create_session(request.session_id.clone()).await;
    
    // Process query with context
    let processed_query = if let Some(ref session_id) = request.session_id {
        enhance_query_with_context(&request.query, &session).await
    } else {
        request.query.clone()
    };

    info!("🧠 Enhanced query: '{}'", processed_query);

    // Generate embedding for the processed query
    let embedding_service = embedding_service.read().await;
    let embedding_result = embedding_service.embed_text(&processed_query)
        .await
        .map_err(|e| {
            warn!("Failed to generate embedding: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    drop(embedding_service);

    // Search for similar vectors
    let limit = request.limit.unwrap_or(10);
    let similar_results = vector_backend.search_similar(&embedding_result.vector, limit * 2)
        .await
        .unwrap_or_default();

    // Process and enhance results with session filtering
    let mut search_results = Vec::new();
    for (id, similarity) in similar_results.into_iter() {
        if let Some(point) = vector_backend.get(&id).await {
            // Filter by session_id if provided
            let matches_session = if let Some(ref session_id) = request.session_id {
                point.metadata.get("session_id").map(|s| s == session_id).unwrap_or(false)
            } else {
                true // No session filter, include all results
            };
            
            if matches_session {
                let result = create_search_result(id, similarity, point, &request.query).await;
                search_results.push(result);
                
                // Stop when we have enough results
                if search_results.len() >= limit {
                    break;
                }
            }
        }
    }

    // Update session
    if let Some(ref session_id) = request.session_id {
        session_manager.update_session(session_id, &request.query, search_results.len()).await;
    }

    // Generate suggestions if requested
    let suggestions = if request.include_suggestions.unwrap_or(true) {
        Some(generate_search_suggestions(&request.query, &search_results).await)
    } else {
        None
    };

    // Create session context
    let session_context = if request.session_id.is_some() {
        Some(create_session_context(&session))
    } else {
        None
    };

    let search_time = start_time.elapsed().as_millis() as u64;

    info!("✅ Semantic search completed: {} results in {}ms", 
          search_results.len(), search_time);

    Ok(Json(SemanticSearchResponse {
        total_found: search_results.len(),
        results: search_results,
        query_processed: processed_query,
        search_time_ms: search_time,
        session_context,
        suggestions,
        facets: None, // TODO: Implement faceted search
        personalization: None, // TODO: Implement personalization
    }))
}

/// Contextual search using session history
pub async fn contextual_search(
    State((vector_backend, embedding_service, session_manager)): State<SearchApiState>,
    Json(request): Json<ContextualSearchRequest>,
) -> Result<Json<SemanticSearchResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    info!("🔍 Contextual search: '{}' (session: {})", 
          request.query, request.session_id);

    // Get session context
    let session = session_manager.get_or_create_session(Some(request.session_id.clone())).await;
    
    // Build contextual query using session history
    let contextual_query = build_contextual_query(&request.query, &session, 
                                                 request.context_depth.unwrap_or(5)).await;
    
    info!("🧠 Contextual query: '{}'", contextual_query);

    // Generate embedding for contextual query
    let embedding_service = embedding_service.read().await;
    let embedding_result = embedding_service.embed_text(&contextual_query)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    drop(embedding_service);

    // Search with enhanced context
    let similar_results = vector_backend.search_similar(&embedding_result.vector, 15)
        .await
        .unwrap_or_default();

    // Process results with context awareness and session filtering
    let mut search_results = Vec::new();
    for (id, similarity) in similar_results.into_iter() {
        if let Some(point) = vector_backend.get(&id).await {
            // Filter by session_id for contextual search (required session_id)
            let matches_session = point.metadata.get("session_id")
                .map(|s| s == &request.session_id).unwrap_or(false);
            
            if matches_session {
                let mut result = create_search_result(id, similarity, point, &request.query).await;
                // Boost relevance if matches session context
                result.score = boost_contextual_relevance(result.score, &result, &session);
                search_results.push(result);
                
                // Stop when we have enough results
                if search_results.len() >= 10 {
                    break;
                }
            }
        }
    }

    // Sort by boosted relevance
    search_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    // Update session
    session_manager.update_session(&request.session_id, &request.query, search_results.len()).await;

    let search_time = start_time.elapsed().as_millis() as u64;

    Ok(Json(SemanticSearchResponse {
        total_found: search_results.len(),
        results: search_results,
        query_processed: contextual_query,
        search_time_ms: search_time,
        session_context: Some(create_session_context(&session)),
        suggestions: Some(generate_contextual_suggestions(&session).await),
        facets: None,
        personalization: None,
    }))
}

/// Get search suggestions for auto-complete
pub async fn get_suggestions(
    State((_vector_backend, _embedding_service, session_manager)): State<SearchApiState>,
    Json(request): Json<SuggestionsRequest>,
) -> Result<Json<SuggestionsResponse>, StatusCode> {
    info!("💭 Generating suggestions for: '{}'", request.partial_query);

    let session = if let Some(session_id) = request.session_id {
        Some(session_manager.get_or_create_session(Some(session_id)).await)
    } else {
        None
    };

    let suggestions = generate_query_suggestions(&request.partial_query, session.as_ref()).await;
    let popular_searches = if request.include_popular.unwrap_or(false) {
        Some(get_popular_searches().await)
    } else {
        None
    };

    Ok(Json(SuggestionsResponse {
        suggestions,
        popular_searches,
        personalized: session.is_some(),
    }))
}

/// Get trending search topics
pub async fn get_trending(
    State((_vector_backend, _embedding_service, _session_manager)): State<SearchApiState>,
    Json(request): Json<TrendingRequest>,
) -> Result<Json<TrendingResponse>, StatusCode> {
    let time_period = request.time_period.unwrap_or_else(|| "24h".to_string());
    info!("📈 Getting trending topics for: {}", time_period);

    // TODO: Implement actual trending analysis
    let trending_topics = vec![
        TrendingTopic {
            topic: "API Authentication".to_string(),
            query_count: 156,
            growth_rate: 23.5,
            related_queries: vec!["JWT tokens".to_string(), "OAuth2".to_string()],
            top_result_domains: vec![SearchDomain::Documents, SearchDomain::Code],
        },
        TrendingTopic {
            topic: "Machine Learning".to_string(),
            query_count: 143,
            growth_rate: 18.2,
            related_queries: vec!["Neural networks".to_string(), "Deep learning".to_string()],
            top_result_domains: vec![SearchDomain::Knowledge, SearchDomain::Documents],
        },
    ];

    Ok(Json(TrendingResponse {
        topics: trending_topics,
        time_period,
        total_queries: 1250,
        generated_at: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Get search analytics for a session
pub async fn get_search_analytics(
    State((_vector_backend, _embedding_service, session_manager)): State<SearchApiState>,
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    info!("📊 Getting analytics for session: {}", session_id);

    let sessions = session_manager.sessions.read().await;
    if let Some(session) = sessions.get(&session_id) {
        Ok(Json(serde_json::json!({
            "session_id": session.session_id,
            "total_queries": session.total_queries,
            "session_duration": chrono::Utc::now().signed_duration_since(session.created_at).num_minutes(),
            "query_history": session.query_history.len(),
            "context_topics": session.context_topics,
            "last_activity": session.last_activity,
            "user_preferences": session.user_preferences
        })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

/// Enhance query with session context
async fn enhance_query_with_context(query: &str, session: &SearchSession) -> String {
    if session.context_topics.is_empty() {
        return query.to_string();
    }
    
    // Simple context enhancement - can be made more sophisticated
    let recent_topics = session.context_topics.iter()
        .rev()
        .take(3)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    
    if !recent_topics.is_empty() && !query.to_lowercase().contains(&recent_topics.to_lowercase()) {
        format!("{} {}", query, recent_topics)
    } else {
        query.to_string()
    }
}

/// Build contextual query using session history
async fn build_contextual_query(query: &str, session: &SearchSession, depth: usize) -> String {
    let mut contextual_parts = vec![query.to_string()];
    
    // Add recent query context
    for entry in session.query_history.iter().rev().take(depth) {
        if !entry.query.is_empty() && entry.query != query {
            contextual_parts.push(entry.query.clone());
        }
    }
    
    contextual_parts.join(" ")
}

/// Create search result from vector point
async fn create_search_result(
    id: Uuid,
    similarity: f32,
    point: crate::vector::VectorPoint,
    original_query: &str,
) -> SearchResult {
    let title = point.metadata.get("title")
        .or_else(|| point.metadata.get("source_text"))
        .cloned()
        .unwrap_or_else(|| "Untitled".to_string());
    
    let content = point.metadata.get("source_text")
        .or_else(|| point.metadata.get("content"))
        .cloned()
        .unwrap_or_else(|| "No content available".to_string());
    
    let content_excerpt = if content.len() > 200 {
        format!("{}...", &content[..197])
    } else {
        content.clone()
    };

    // Simple highlighting (can be enhanced)
    let highlights = create_text_highlights(&content_excerpt, original_query);
    
    SearchResult {
        id,
        title: if title.len() > 100 { format!("{}...", &title[..97]) } else { title },
        content: content_excerpt,
        url: point.metadata.get("url").cloned(),
        score: similarity,
        domain: detect_search_domain(&point.metadata),
        metadata: point.metadata,
        highlights,
        related_results: Vec::new(), // TODO: Implement related results
        timestamp: None, // TODO: Extract timestamp
        author: None, // TODO: Extract author
    }
}

/// Create text highlights for search terms
fn create_text_highlights(content: &str, query: &str) -> Vec<TextHighlight> {
    let mut highlights = Vec::new();
    let query_words: Vec<&str> = query.split_whitespace().collect();
    
    for word in query_words {
        if word.len() > 2 {
            if let Some(pos) = content.to_lowercase().find(&word.to_lowercase()) {
                highlights.push(TextHighlight {
                    text: word.to_string(),
                    start: pos,
                    end: pos + word.len(),
                    relevance: 1.0,
                });
            }
        }
    }
    
    highlights
}

/// Detect search domain from metadata
fn detect_search_domain(metadata: &HashMap<String, String>) -> SearchDomain {
    if let Some(category) = metadata.get("category") {
        match category.as_str() {
            "conversation" => SearchDomain::Conversations,
            "document" | "article" => SearchDomain::Documents,
            "code" | "programming" => SearchDomain::Code,
            "knowledge" => SearchDomain::Knowledge,
            _ => SearchDomain::All,
        }
    } else {
        SearchDomain::All
    }
}

/// Generate search suggestions
async fn generate_search_suggestions(
    query: &str,
    results: &[SearchResult],
) -> Vec<SearchSuggestion> {
    let mut suggestions = Vec::new();
    
    // Related concept suggestions based on results
    let mut concepts: HashMap<String, usize> = HashMap::new();
    for result in results.iter().take(5) {
        for word in result.title.split_whitespace() {
            if word.len() > 3 && !query.to_lowercase().contains(&word.to_lowercase()) {
                *concepts.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }
    }
    
    // Create suggestions from frequent concepts
    for (concept, count) in concepts.iter() {
        if *count >= 2 {
            suggestions.push(SearchSuggestion {
                query: format!("{} {}", query, concept),
                type_: SuggestionType::Related,
                confidence: (*count as f32) / results.len() as f32,
                preview_count: *count,
            });
        }
    }
    
    suggestions.truncate(5);
    suggestions
}

/// Generate contextual suggestions based on session
async fn generate_contextual_suggestions(session: &SearchSession) -> Vec<SearchSuggestion> {
    let mut suggestions = Vec::new();
    
    // Suggest based on recent topics
    for topic in session.context_topics.iter().rev().take(3) {
        suggestions.push(SearchSuggestion {
            query: format!("More about {}", topic),
            type_: SuggestionType::Drill,
            confidence: 0.8,
            preview_count: 0,
        });
    }
    
    suggestions
}

/// Generate query suggestions for auto-complete
async fn generate_query_suggestions(
    partial_query: &str,
    session: Option<&SearchSession>,
) -> Vec<SearchSuggestion> {
    let mut suggestions = Vec::new();
    
    // Simple completion suggestions
    let completions = vec![
        "how to implement",
        "what is",
        "best practices for",
        "tutorial on",
        "examples of",
    ];
    
    for completion in completions {
        if completion.starts_with(partial_query) || partial_query.len() < 3 {
            suggestions.push(SearchSuggestion {
                query: completion.to_string(),
                type_: SuggestionType::Completion,
                confidence: 0.7,
                preview_count: 0,
            });
        }
    }
    
    // Add personalized suggestions from session
    if let Some(session) = session {
        for entry in session.query_history.iter().rev().take(3) {
            if entry.query.to_lowercase().contains(&partial_query.to_lowercase()) {
                suggestions.push(SearchSuggestion {
                    query: entry.query.clone(),
                    type_: SuggestionType::Related,
                    confidence: 0.9,
                    preview_count: entry.results_count,
                });
            }
        }
    }
    
    suggestions.truncate(5);
    suggestions
}

/// Get popular searches (mock implementation)
async fn get_popular_searches() -> Vec<String> {
    vec![
        "API authentication".to_string(),
        "machine learning basics".to_string(),
        "REST API design".to_string(),
        "database optimization".to_string(),
        "JavaScript tutorials".to_string(),
    ]
}

/// Create session context
fn create_session_context(session: &SearchSession) -> SessionContext {
    let recent_queries = session.query_history
        .iter()
        .rev()
        .take(5)
        .map(|entry| entry.query.clone())
        .collect();

    SessionContext {
        session_id: session.session_id.clone(),
        query_history: recent_queries,
        context_topics: session.context_topics.clone(),
        search_intent: detect_search_intent(session),
        personalization_score: calculate_personalization_score(session),
    }
}

/// Detect search intent from session
fn detect_search_intent(session: &SearchSession) -> Option<SearchIntent> {
    let recent_queries: String = session.query_history
        .iter()
        .rev()
        .take(3)
        .map(|e| e.query.clone())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();

    if recent_queries.contains("how to") || recent_queries.contains("implement") {
        Some(SearchIntent::Implementation)
    } else if recent_queries.contains("error") || recent_queries.contains("fix") {
        Some(SearchIntent::Troubleshooting)
    } else if recent_queries.contains("what is") || recent_queries.contains("explain") {
        Some(SearchIntent::Learning)
    } else if recent_queries.contains("vs") || recent_queries.contains("compare") {
        Some(SearchIntent::Comparison)
    } else {
        Some(SearchIntent::Exploration)
    }
}

/// Calculate personalization score
fn calculate_personalization_score(session: &SearchSession) -> f32 {
    let base_score = (session.total_queries as f32).min(20.0) / 20.0;
    let topic_diversity = (session.context_topics.len() as f32).min(10.0) / 10.0;
    (base_score + topic_diversity) / 2.0
}

/// Boost contextual relevance
fn boost_contextual_relevance(
    original_score: f32,
    result: &SearchResult,
    session: &SearchSession,
) -> f32 {
    let mut boosted_score = original_score;
    
    // Boost if matches session topics
    for topic in &session.context_topics {
        if result.title.to_lowercase().contains(&topic.to_lowercase()) ||
           result.content.to_lowercase().contains(&topic.to_lowercase()) {
            boosted_score += 0.1;
        }
    }
    
    boosted_score.min(1.0)
}

/// Extract keywords from text (simple implementation)
fn extract_keywords(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter(|word| word.len() > 3)
        .map(|word| word.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .filter(|word| !word.is_empty())
        .collect()
}

// ================================================================================================
// ROUTER CREATION
// ================================================================================================

/// Create the semantic search API router
pub fn create_search_router() -> Router<SearchApiState> {
    Router::new()
        .route("/api/v1/search/semantic", post(semantic_search))
        .route("/api/v1/search/contextual", post(contextual_search))
        .route("/api/v1/search/suggest", post(get_suggestions))
        .route("/api/v1/search/trending", post(get_trending))
        .route("/api/v1/search/analytics/{session_id}", get(get_search_analytics))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_domain_detection() {
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "conversation".to_string());
        assert!(matches!(detect_search_domain(&metadata), SearchDomain::Conversations));
        
        metadata.insert("category".to_string(), "code".to_string());
        assert!(matches!(detect_search_domain(&metadata), SearchDomain::Code));
    }

    #[test]
    fn test_keyword_extraction() {
        let keywords = extract_keywords("How to implement REST API authentication");
        assert!(keywords.contains(&"implement".to_string()));
        assert!(keywords.contains(&"authentication".to_string()));
        assert!(!keywords.contains(&"to".to_string())); // Short word filtered
    }

    #[tokio::test]
    async fn test_query_enhancement() {
        let session = SearchSession {
            session_id: "test".to_string(),
            query_history: Vec::new(),
            context_topics: vec!["REST".to_string(), "API".to_string()],
            user_preferences: HashMap::new(),
            created_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            total_queries: 0,
        };
        
        let enhanced = enhance_query_with_context("authentication", &session).await;
        assert!(enhanced.contains("authentication"));
        assert!(enhanced.contains("REST") || enhanced.contains("API"));
    }
}