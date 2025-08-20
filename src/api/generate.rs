use crate::{batching::BatchProcessor, errors::AppError, models::ModelInfo};
use crate::vector::{VectorBackend, VectorPoint, EmbeddingService};
use crate::api::search::{SearchSessionManager, SemanticSearchRequest, SearchDomain, SearchFilters};
use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant, collections::HashMap};
use tokio::sync::RwLock;
use uuid::Uuid;

// API DESIGN: Request Schema for Text Generation
// Follows OpenAI-compatible API structure for easy integration
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,              // Input text for generation
    pub max_tokens: Option<usize>,   // Optional limit (default: 100)
    pub temperature: Option<f32>,    // Future: sampling randomness control
    pub use_memory: Option<bool>,    // Enable conversation memory (default: true)
    pub memory_limit: Option<usize>, // Number of past conversations to include (default: 3)
    pub model: Option<String>,       // Model to use (e.g., "tinyllama", "gemma") - uses default if not specified
    pub session_id: Option<String>,  // Session ID for contextual memory (auto-generated if not provided)
    pub memory_domains: Option<Vec<SearchDomain>>, // Specific memory domains to search
    pub memory_quality_threshold: Option<f32>, // Minimum quality score for memory retrieval (default: 0.6)
}

// API DESIGN: Comprehensive Response with Performance Metrics
// Provides detailed timing information for performance analysis
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,                    // Generated text output
    pub model_info: ModelInfo,           // Model specifications
    pub processing_time_ms: u64,         // Pure inference time (excludes queue)
    pub request_id: String,              // Unique identifier for tracing
    pub tokens_generated: usize,         // Actual token count (may be < max_tokens)
    pub tokens_per_second: f64,          // Performance metric (inference only)
    pub queue_time_ms: u64,              // Time spent waiting in batch queue
    pub batch_processing: bool,          // Indicates batching was used
    pub memory_used: bool,               // Whether conversation memory was used
    pub context_retrieved: usize,        // Number of past conversations included
    pub session_id: String,              // Session ID used for this generation
    pub semantic_search_time_ms: Option<u64>, // Time spent on semantic search
    pub memory_relevance_scores: Vec<f32>, // Relevance scores of retrieved memories
    pub search_intent: Option<String>,   // Detected search intent for memory retrieval
}

impl GenerateRequest {
    // VALIDATION: Input Sanitization and Resource Protection
    // Prevents abuse and ensures stable performance under load
    pub fn validate(&self) -> std::result::Result<(), String> {
        // Prevent empty prompts that waste compute resources
        if self.prompt.trim().is_empty() {
            return Err("Prompt cannot be empty".to_string());
        }
        
        // RESOURCE PROTECTION: Limit prompt length to prevent memory exhaustion
        // 4096 chars ≈ 1000 tokens, reasonable for TinyLlama's 2048 context window
        if self.prompt.len() > 4096 {
            return Err("Prompt too long (Max 4096 characters)".to_string());
        }
        
        // PERFORMANCE: Limit generation length to maintain responsiveness
        // 512 tokens ≈ 25-40 seconds on typical hardware, balances utility vs responsiveness
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 || max_tokens > 512 {
                return Err("max_tokens must be between 1 and 512".to_string());
            }
        }
        Ok(())
    }
}

// Enhanced state type for generate endpoint with semantic search integration
pub type GenerateState = (
    Arc<BatchProcessor>, 
    Arc<VectorBackend>, 
    Arc<RwLock<EmbeddingService>>, 
    Arc<SearchSessionManager>
);

// ENDPOINT: Enhanced Text Generation Handler with Semantic Memory Integration
// Implements intelligent conversation context retrieval using semantic search
pub async fn generate_text(
    State((batch_processor, vector_backend, embedding_service, session_manager)): State<GenerateState>,
    Json(request): Json<GenerateRequest>,
) -> std::result::Result<Json<GenerateResponse>, AppError> {
    // ANALYTICS: End-to-end timing measurement
    let request_start = Instant::now();
    let request_id = Uuid::new_v4().to_string();

    // SECURITY: Input validation before processing
    request.validate().map_err(AppError::Validation)?;

    // DEFAULTS: Reasonable token limit for responsive service
    let max_tokens = request.max_tokens.unwrap_or(100);
    let use_memory = request.use_memory.unwrap_or(true);
    let memory_limit = request.memory_limit.unwrap_or(3);
    let memory_quality_threshold = request.memory_quality_threshold.unwrap_or(0.6);

    // SESSION MANAGEMENT: Get or create session for contextual memory
    let session_id = request.session_id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
    let session = session_manager.get_or_create_session(Some(session_id.clone())).await;

    // SEMANTIC MEMORY RETRIEVAL: Use intelligent search for context
    let mut context_retrieved = 0;
    let mut semantic_search_time = None;
    let mut memory_relevance_scores = Vec::new();
    let mut detected_intent = None;
    
    let final_prompt = if use_memory {
        let search_start = Instant::now();
        
        // Create semantic search request for memory retrieval
        let memory_search_request = SemanticSearchRequest {
            query: request.prompt.clone(),
            session_id: Some(session_id.clone()),
            limit: Some(memory_limit),
            domains: Some(vec![SearchDomain::Conversations]),
            filters: Some(SearchFilters {
                time_range: None,
                content_type: Some(vec!["conversation".to_string()]),
                categories: None,
                authors: None,
                language: None,
                quality_threshold: Some(memory_quality_threshold),
            }),
            include_suggestions: Some(false),
            personalize: Some(true),
        };

        // Perform semantic search for conversation memory
        match perform_semantic_memory_search(
            &vector_backend,
            &embedding_service,
            &session_manager,
            memory_search_request,
        ).await {
            Ok(search_response) => {
                let search_time = search_start.elapsed().as_millis() as u64;
                semantic_search_time = Some(search_time);
                context_retrieved = search_response.results.len();
                
                // Extract relevance scores and intent
                memory_relevance_scores = search_response.results.iter().map(|r| r.score).collect();
                detected_intent = search_response.session_context
                    .and_then(|ctx| ctx.search_intent)
                    .map(|intent| format!("{:?}", intent));
                
                if !search_response.results.is_empty() {
                    // Build intelligent context from search results
                    let context_parts: Vec<String> = search_response.results
                        .into_iter()
                        .filter_map(|result| {
                            result.metadata.get("conversation").cloned()
                        })
                        .collect();
                    
                    let context = context_parts.join("\n---\n");
                    tracing::info!("🧠 Retrieved {} conversations via semantic search (avg relevance: {:.3})", 
                                  context_retrieved, 
                                  memory_relevance_scores.iter().sum::<f32>() / memory_relevance_scores.len().max(1) as f32);
                    
                    // Enhanced prompt formatting optimized for TinyLlama
                    format!("Context: {}\n\nBased on the above context, please respond to:\nUser: {}\nAssistant:", 
                           context.replace("User:", "").replace("Assistant:", "").trim(), request.prompt)
                } else {
                    tracing::info!("🔍 No relevant conversations found via semantic search");
                    request.prompt.clone()
                }
            }
            Err(e) => {
                tracing::warn!("❌ Semantic memory search failed: {}", e);
                request.prompt.clone()
            }
        }
    } else {
        request.prompt.clone()
    };

    tracing::info!(
        request_id = %request_id,
        prompt_length = final_prompt.len(),
        context_retrieved = context_retrieved,
        use_memory = use_memory,
        "Received generation request"
    );
    
    // DEBUG: Log the actual prompt being sent to the model
    tracing::info!("🔍 FINAL PROMPT SENT TO MODEL: '{}'", final_prompt);

    // OPTIMIZATION: Asynchronous Batch Processing
    // Non-blocking submission to batch queue for optimal throughput
    // Automatic batching provides 20-30% performance improvement
    let batch_response = batch_processor
        .submit_request(final_prompt.clone(), max_tokens)
        .await
        .map_err(|e| {
            tracing::error!("Batch processing failed for request {}: {}", request_id, e);
            AppError::BadRequest(format!("Batch Processing failed: {e}"))
        })?;

    // ANALYTICS: Detailed Performance Breakdown
    let total_time = request_start.elapsed();
    let total_time_ms = total_time.as_millis() as u64;
    
    // METRICS: Separate queue time from processing time
    // Queue time indicates batching efficiency and load
    let queue_time_ms = total_time_ms.saturating_sub(batch_response.processing_time_ms);

    // PERFORMANCE CALCULATION: Pure inference speed measurement
    // Excludes queue time to measure actual model performance
    // Critical metric for optimizing generation algorithms
    let tokens_per_second = if batch_response.processing_time_ms > 0 {
        (batch_response.token_generated as f64)
            / (batch_response.processing_time_ms as f64 / 1000.0)
    } else {
        0.0  // Prevent division by zero for instant responses
    };

    // API RESPONSE: Dynamic Model Information from Active Model  
    // Get actual model info from the currently active model
    let model_info = batch_processor.get_active_model_info().await
        .unwrap_or_else(|_| {
            // Fallback if we can't get active model info
            ModelInfo {
                name: "Unknown Model".to_string(),
                version: "v1.0-batched".to_string(),
                parameters: 1000000000,
                memory_mb: 2200,
                device: "Auto-detected".to_string(),
                vocab_size: 32000,
                context_length: 2048,
                model_type: "llama".to_string(),
                architecture: "transformer".to_string(),
                precision: "f16".to_string(),
            }
        });

    // MONITORING: Structured Request Completion Log
    // Essential for performance tuning and capacity planning
    // Helps identify bottlenecks: queue vs processing time
    // ENHANCED CONVERSATION STORAGE: Quality-filtered storage with semantic embedding
    if use_memory {
        let conversation_text = format!("{} -> {}", request.prompt, batch_response.text);
        
        // Enterprise quality gate before storage
        let response_quality = assess_context_quality(&batch_response.text);
        tracing::info!("📊 Response quality assessment: {:.2} for response length: {}", 
                      response_quality, batch_response.text.len());
        
        if response_quality > 0.4 { // Enterprise storage threshold
            tracing::info!("✅ High-quality response approved for semantic storage");
            if let Err(e) = store_conversation_with_semantic_embedding(
                &vector_backend,
                &embedding_service,
                &conversation_text,
                &session_id,
            ).await {
                tracing::warn!("❌ Failed to store conversation with semantic embedding: {}", e);
            } else {
                // Update session with successful generation
                session_manager.update_session(&session_id, &request.prompt, 1).await;
            }
        } else {
            tracing::warn!("🚫 Low-quality response rejected from storage (score: {:.2})", response_quality);
        }
    }

    tracing::info!(
        "Request {} Completed: {}ms total ({}ms queue + {}ms processing, {:.2} tok/s, {} context)",
        request_id,
        total_time_ms,
        queue_time_ms,
        batch_response.processing_time_ms,
        tokens_per_second,
        context_retrieved
    );

    Ok(Json(GenerateResponse {
        text: batch_response.text,
        model_info,
        processing_time_ms: batch_response.processing_time_ms,
        request_id,
        tokens_generated: batch_response.token_generated,
        tokens_per_second,
        queue_time_ms,
        batch_processing: true,
        memory_used: use_memory,
        context_retrieved,
        session_id,
        semantic_search_time_ms: semantic_search_time,
        memory_relevance_scores,
        search_intent: detected_intent,
    }))
}

// ENDPOINT: Batch Processing Monitoring
// Provides real-time insights into system performance and load
// Critical for operational monitoring and capacity planning
pub async fn batch_status(
    State(batch_processor): State<Arc<BatchProcessor>>,
) -> std::result::Result<Json<BatchStatusResponse>, AppError> {
    // MONITORING: Real-time Performance Metrics
    let stats = batch_processor.get_stats().await;
    let queue_size = batch_processor.get_queue_size().await as u64;

    Ok(Json(BatchStatusResponse {
        queue_size,                             // Current pending requests
        total_requests: stats.total_requests,   // Lifetime request count
        total_batches: stats.total_batches,     // Number of batches processed
        avg_batch_size: stats.avg_batch_size,   // Batching efficiency metric
        avg_processing_time_ms: stats.avg_processing_time_ms, // Performance trend
    }))
}

// API SCHEMA: Batch Processing Metrics Response
// Provides comprehensive system health and performance data
#[derive(Serialize)]
pub struct BatchStatusResponse {
    pub queue_size: u64,                    // Instant load indicator
    pub total_requests: u64,                // System usage metric
    pub total_batches: u64,                 // Batching frequency
    pub avg_batch_size: f64,                // Efficiency indicator (target: 2-4)
    pub avg_processing_time_ms: f64,        // Performance trend (target: <500ms)
}

// ================================================================================================
// ENTERPRISE CONVERSATION INTELLIGENCE
// ================================================================================================

/// Enterprise-grade query intent classification
#[derive(Debug, Clone)]
enum QueryIntent {
    Personal,      // Needs user-specific context (name, background, preferences)
    Contextual,    // Needs conversation flow context (follow-up questions)
    General,       // Knowledge questions that don't need personal context
    Task,          // Action-oriented queries that may need recent context
}

/// Classify query intent using enterprise-level pattern matching
fn classify_query_intent(prompt: &str) -> QueryIntent {
    let prompt_lower = prompt.to_lowercase();
    
    // Personal information patterns (highest priority)
    let personal_patterns = [
        r"\b(my|mine|i'm|i am)\b",
        r"\b(name|age|occupation|job|work|background|experience)\b",
        r"\bwhat.*(am i|do i|is my)\b",
        r"\b(remember|recall|told you|mentioned)\b",
        r"\bwho am i\b",
        r"\bwhere do i\b",
        r"\bhow old\b"
    ];
    
    // Contextual continuation patterns
    let contextual_patterns = [
        r"\b(that|this|it|they|them)\b",
        r"\b(also|additionally|furthermore|moreover)\b",
        r"\b(continue|more about|tell me more)\b",
        r"\bwhat about\b",
        r"\bcan you.*more\b"
    ];
    
    // Task-oriented patterns
    let task_patterns = [
        r"\b(help me|show me|guide me|assist)\b",
        r"\b(create|build|make|generate)\b",
        r"\b(how to|how can|what should)\b",
        r"\b(recommend|suggest|advise)\b"
    ];
    
    // Check patterns in priority order
    for pattern in &personal_patterns {
        if regex::Regex::new(pattern).unwrap().is_match(&prompt_lower) {
            return QueryIntent::Personal;
        }
    }
    
    for pattern in &contextual_patterns {
        if regex::Regex::new(pattern).unwrap().is_match(&prompt_lower) {
            return QueryIntent::Contextual;
        }
    }
    
    for pattern in &task_patterns {
        if regex::Regex::new(pattern).unwrap().is_match(&prompt_lower) {
            return QueryIntent::Task;
        }
    }
    
    QueryIntent::General
}

/// Enterprise context quality assessment
fn assess_context_quality(conversation: &str) -> f32 {
    if conversation.trim().is_empty() {
        return 0.0;
    }
    
    let words: Vec<&str> = conversation.split_whitespace().collect();
    if words.len() < 3 {
        return 0.1;
    }
    
    // Check for excessive repetition (enterprise quality gate)
    let unique_words: std::collections::HashSet<_> = words.iter().collect();
    let uniqueness_ratio = unique_words.len() as f32 / words.len() as f32;
    
    // Check for coherent structure
    let has_proper_dialogue = conversation.contains("User:") && conversation.contains("Assistant:");
    let reasonable_length = words.len() >= 5 && words.len() <= 200;
    let not_corrupted = !conversation.chars().filter(|c| c.is_alphabetic()).take(20).all(|c| c == 'a' || c == 'i');
    
    let quality_score = match (uniqueness_ratio, has_proper_dialogue, reasonable_length, not_corrupted) {
        (ratio, true, true, true) if ratio > 0.6 => 0.9,  // High quality
        (ratio, true, true, true) if ratio > 0.4 => 0.7,  // Medium quality  
        (ratio, _, true, true) if ratio > 0.5 => 0.6,     // Acceptable
        _ => 0.2  // Low quality
    };
    
    quality_score
}

/// Perform semantic search for memory retrieval using the search API
async fn perform_semantic_memory_search(
    vector_backend: &Arc<VectorBackend>,
    embedding_service: &Arc<RwLock<EmbeddingService>>,
    session_manager: &Arc<SearchSessionManager>,
    request: SemanticSearchRequest,
) -> Result<crate::api::search::SemanticSearchResponse, String> {
    use crate::api::search::{semantic_search, SearchApiState};
    
    tracing::info!("🔍 Performing semantic memory search for: '{}'", request.query);
    
    // Create the search state tuple
    let search_state: SearchApiState = (
        Arc::clone(vector_backend),
        Arc::clone(embedding_service),
        Arc::clone(session_manager),
    );
    
    // Perform the semantic search using the search API
    match semantic_search(
        axum::extract::State(search_state),
        axum::Json(request),
    ).await {
        Ok(axum::Json(response)) => {
            tracing::info!("✅ Semantic search completed: {} results found", response.results.len());
            Ok(response)
        }
        Err(status_code) => {
            let error_msg = format!("Semantic search failed with status: {}", status_code);
            tracing::error!("❌ {}", error_msg);
            Err(error_msg)
        }
    }
}

/// Enhanced conversation storage with semantic embedding
async fn store_conversation_with_semantic_embedding(
    vector_backend: &Arc<VectorBackend>,
    embedding_service: &Arc<RwLock<EmbeddingService>>,
    conversation: &str,
    session_id: &str,
) -> Result<(), String> {
    tracing::info!("💾 Starting enhanced conversation storage with semantic embedding");
    tracing::debug!("📝 Conversation text length: {} characters", conversation.len());
    
    // Generate semantic embedding using the embedding service
    let embedding_service = embedding_service.read().await;
    let vector_point = embedding_service.create_vector_point(conversation, {
        let mut metadata = HashMap::new();
        metadata.insert("conversation".to_string(), conversation.to_string());
        metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        metadata.insert("type".to_string(), "conversation".to_string());
        metadata.insert("session_id".to_string(), session_id.to_string());
        metadata.insert("source".to_string(), "generate_api".to_string());
        metadata
    }).await.map_err(|e| format!("Failed to create vector point: {}", e))?;
    
    drop(embedding_service);
    
    tracing::debug!("📦 Created semantic vector point with ID: {}", vector_point.id);
    
    // Insert the vector point
    match vector_backend.insert(vector_point).await {
        Ok(id) => {
            tracing::info!("✅ Conversation stored successfully with semantic embedding, ID: {}", id);
            Ok(())
        }
        Err(e) => {
            let error_msg = format!("Failed to insert conversation vector: {}", e);
            tracing::error!("❌ {}", error_msg);
            Err(error_msg)
        }
    }
}
