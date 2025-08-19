// ================================================================================================
// ENHANCED GENERATE API - WITH VECTOR STORAGE AND CONVERSATION MEMORY
// ================================================================================================
//
// This enhanced version shows how the generate API stores conversations and uses vector memory
// to provide personalized, context-aware responses.
//
// ================================================================================================

use crate::{
    batching::BatchProcessor, 
    errors::AppError, 
    models::ModelInfo,
    vector::VectorOperations,
    embedding::{EmbeddingProcessor, EmbeddingRequest, create_embedding_from_generation}
};
use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant, collections::HashMap};
use uuid::Uuid;
use serde_json::json;

// ================================================================================================
// ENHANCED REQUEST SCHEMA
// ================================================================================================

#[derive(Debug, Deserialize)]
pub struct EnhancedGenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    
    // NEW: Vector storage and personalization options
    pub user_id: Option<String>,                    // Track user for personalization
    pub session_id: Option<String>,                 // Group related conversations
    pub store_conversation: Option<bool>,           // Whether to save this conversation
    pub use_context: Option<bool>,                  // Use previous conversations as context
    pub context_limit: Option<usize>,               // How many past conversations to include
    pub collection: Option<String>,                 // Which collection to store/search in
}

#[derive(Debug, Serialize)]
pub struct EnhancedGenerateResponse {
    // Original response fields
    pub text: String,
    pub model_info: ModelInfo,
    pub processing_time_ms: u64,
    pub request_id: String,
    pub tokens_generated: usize,
    pub tokens_per_second: f64,
    pub queue_time_ms: u64,
    pub batch_processing: bool,
    
    // NEW: Vector storage and context fields
    pub conversation_id: Option<String>,            // ID of stored conversation
    pub used_context: Vec<ContextItem>,             // Which past conversations were used
    pub embedding_stored: bool,                     // Whether conversation was saved
    pub similar_conversations: Vec<SimilarConversation>, // Related past conversations
    pub personalization_applied: bool,              // Whether response was personalized
}

#[derive(Debug, Serialize)]
pub struct ContextItem {
    pub conversation_id: String,
    pub summary: String,
    pub relevance_score: f64,
    pub timestamp: String,
}

#[derive(Debug, Serialize)]
pub struct SimilarConversation {
    pub id: String,
    pub prompt: String,
    pub response_summary: String,
    pub similarity_score: f64,
    pub timestamp: String,
}

// ================================================================================================
// ENHANCED STATE - NOW INCLUDES VECTOR OPERATIONS
// ================================================================================================

pub type EnhancedGenerateState = (
    Arc<BatchProcessor>,           // Original batch processor
    Arc<VectorOperations>,         // NEW: Vector database operations
    Arc<EmbeddingProcessor>,       // NEW: Embedding generation
);

// ================================================================================================
// ENHANCED GENERATE ENDPOINT
// ================================================================================================

pub async fn enhanced_generate_text(
    State((batch_processor, vector_ops, embedding_processor)): State<EnhancedGenerateState>,
    Json(request): Json<EnhancedGenerateRequest>,
) -> std::result::Result<Json<EnhancedGenerateResponse>, AppError> {
    let request_start = Instant::now();
    let request_id = Uuid::new_v4().to_string();
    
    // Validate request
    validate_enhanced_request(&request)?;
    
    let max_tokens = request.max_tokens.unwrap_or(100);
    let use_context = request.use_context.unwrap_or(false);
    let store_conversation = request.store_conversation.unwrap_or(true);
    let collection = request.collection.as_deref().unwrap_or("conversations");
    
    tracing::info!(
        request_id = %request_id,
        user_id = ?request.user_id,
        session_id = ?request.session_id,
        use_context = use_context,
        store_conversation = store_conversation,
        "Enhanced generation request received"
    );
    
    // ============================================================================================
    // STEP 1: RETRIEVE CONTEXT FROM PAST CONVERSATIONS (NEW!)
    // ============================================================================================
    
    let mut context_items = Vec::new();
    let mut enhanced_prompt = request.prompt.clone();
    let mut personalization_applied = false;
    
    if use_context && request.user_id.is_some() {
        match retrieve_user_context(&vector_ops, &embedding_processor, &request).await {
            Ok((context, enhanced)) => {
                context_items = context;
                if !context_items.is_empty() {
                    // Add context to the prompt
                    let context_text = context_items.iter()
                        .map(|c| format!("Previous conversation: {}", c.summary))
                        .collect::<Vec<_>>()
                        .join("\n");
                    
                    enhanced_prompt = format!(
                        "Context from previous conversations:\n{}\n\nCurrent question: {}", 
                        context_text, 
                        request.prompt
                    );
                    personalization_applied = true;
                    
                    tracing::info!(
                        request_id = %request_id,
                        context_items = context_items.len(),
                        "Added context from previous conversations"
                    );
                }
            },
            Err(e) => {
                tracing::warn!("Failed to retrieve context: {}", e);
                // Continue without context
            }
        }
    }
    
    // ============================================================================================
    // STEP 2: GENERATE RESPONSE USING ENHANCED PROMPT
    // ============================================================================================
    
    let batch_response = batch_processor
        .submit_request(enhanced_prompt, max_tokens)
        .await
        .map_err(|e| {
            tracing::error!("Batch processing failed for request {}: {}", request_id, e);
            AppError::BadRequest(format!("Batch Processing failed: {e}"))
        })?;
    
    // ============================================================================================
    // STEP 3: STORE CONVERSATION AS VECTOR (NEW!)
    // ============================================================================================
    
    let mut conversation_id = None;
    let mut embedding_stored = false;
    let mut similar_conversations = Vec::new();
    
    if store_conversation {
        match store_conversation_vector(
            &embedding_processor,
            &vector_ops,
            &request,
            &batch_response.text,
            &request_id,
            collection
        ).await {
            Ok((conv_id, similar)) => {
                conversation_id = Some(conv_id);
                embedding_stored = true;
                similar_conversations = similar;
                
                tracing::info!(
                    request_id = %request_id,
                    conversation_id = ?conversation_id,
                    similar_found = similar_conversations.len(),
                    "Conversation stored successfully"
                );
            },
            Err(e) => {
                tracing::warn!("Failed to store conversation: {}", e);
                // Continue without storing
            }
        }
    }
    
    // ============================================================================================
    // STEP 4: CALCULATE METRICS AND PREPARE RESPONSE
    // ============================================================================================
    
    let total_time = request_start.elapsed();
    let total_time_ms = total_time.as_millis() as u64;
    let queue_time_ms = total_time_ms.saturating_sub(batch_response.processing_time_ms);
    
    let tokens_per_second = if batch_response.processing_time_ms > 0 {
        (batch_response.token_generated as f64) / (batch_response.processing_time_ms as f64 / 1000.0)
    } else {
        0.0
    };
    
    let model_info = ModelInfo {
        name: "TinyLlama-1.1B-Chat-Enhanced".to_string(),
        version: "v1.0-vector-enabled".to_string(),
        parameters: 110000000,
        memory_mb: 2200,
        device: "Auto-detected".to_string(),
        vocab_size: 32000,
        context_length: 2048,
    };
    
    tracing::info!(
        "Enhanced request {} completed: {}ms total, context_items: {}, stored: {}, personalized: {}",
        request_id,
        total_time_ms,
        context_items.len(),
        embedding_stored,
        personalization_applied
    );
    
    Ok(Json(EnhancedGenerateResponse {
        text: batch_response.text,
        model_info,
        processing_time_ms: batch_response.processing_time_ms,
        request_id,
        tokens_generated: batch_response.token_generated,
        tokens_per_second,
        queue_time_ms,
        batch_processing: true,
        conversation_id,
        used_context: context_items,
        embedding_stored,
        similar_conversations,
        personalization_applied,
    }))
}

// ================================================================================================
// HELPER FUNCTIONS
// ================================================================================================

fn validate_enhanced_request(request: &EnhancedGenerateRequest) -> Result<(), AppError> {
    if request.prompt.trim().is_empty() {
        return Err(AppError::Validation("Prompt cannot be empty".to_string()));
    }
    
    if request.prompt.len() > 4096 {
        return Err(AppError::Validation("Prompt too long (Max 4096 characters)".to_string()));
    }
    
    if let Some(max_tokens) = request.max_tokens {
        if max_tokens == 0 || max_tokens > 512 {
            return Err(AppError::Validation("max_tokens must be between 1 and 512".to_string()));
        }
    }
    
    if let Some(context_limit) = request.context_limit {
        if context_limit > 10 {
            return Err(AppError::Validation("context_limit cannot exceed 10".to_string()));
        }
    }
    
    Ok(())
}

async fn retrieve_user_context(
    vector_ops: &VectorOperations,
    embedding_processor: &EmbeddingProcessor,
    request: &EnhancedGenerateRequest,
) -> Result<(Vec<ContextItem>, String), Box<dyn std::error::Error>> {
    let mut context_items = Vec::new();
    
    if let Some(user_id) = &request.user_id {
        // Generate embedding for current prompt to find similar conversations
        let query_embedding = embedding_processor.generate_embedding(&request.prompt)?;
        
        // Search for similar past conversations
        let search_params = crate::vector::SearchParams::new(query_embedding, request.context_limit.unwrap_or(3))
            .with_score_threshold(0.7)
            .with_filter({
                let mut filter = HashMap::new();
                filter.insert("user_id".to_string(), json!(user_id));
                filter
            });
        
        let collection = request.collection.as_deref().unwrap_or("conversations");
        let search_result = vector_ops.search_vectors(search_params, Some(collection)).await?;
        
        for result in search_result.results {
            if let Some(summary) = result.get_metadata::<String>("conversation_summary") {
                context_items.push(ContextItem {
                    conversation_id: result.id.to_string(),
                    summary,
                    relevance_score: result.score as f64,
                    timestamp: result.get_metadata::<String>("timestamp")
                        .unwrap_or_else(|| "unknown".to_string()),
                });
            }
        }
    }
    
    Ok((context_items, String::new()))
}

async fn store_conversation_vector(
    embedding_processor: &EmbeddingProcessor,
    vector_ops: &VectorOperations,
    request: &EnhancedGenerateRequest,
    response_text: &str,
    request_id: &str,
    collection: &str,
) -> Result<(String, Vec<SimilarConversation>), Box<dyn std::error::Error>> {
    
    // Create metadata for the conversation
    let mut metadata = HashMap::new();
    metadata.insert("request_id".to_string(), json!(request_id));
    metadata.insert("prompt".to_string(), json!(request.prompt));
    metadata.insert("response".to_string(), json!(response_text));
    metadata.insert("timestamp".to_string(), json!(chrono::Utc::now()));
    metadata.insert("conversation_summary".to_string(), json!(format!(
        "Q: {} A: {}", 
        request.prompt.chars().take(100).collect::<String>(),
        response_text.chars().take(100).collect::<String>()
    )));
    
    if let Some(user_id) = &request.user_id {
        metadata.insert("user_id".to_string(), json!(user_id));
    }
    
    if let Some(session_id) = &request.session_id {
        metadata.insert("session_id".to_string(), json!(session_id));
    }
    
    // Generate and store embedding
    let embedding_request = EmbeddingRequest {
        text: format!("{} {}", request.prompt, response_text),
        response: Some(response_text.to_string()),
        context: Some(metadata),
        collection: Some(collection.to_string()),
        store: true,
    };
    
    let embedding_result = embedding_processor.generate_and_store(embedding_request).await?;
    
    // Find similar conversations
    let similar_conversations = find_similar_conversations(
        vector_ops,
        &embedding_result.embedding,
        collection,
        &embedding_result.id
    ).await.unwrap_or_default();
    
    Ok((embedding_result.id.to_string(), similar_conversations))
}

async fn find_similar_conversations(
    vector_ops: &VectorOperations,
    query_embedding: &[f32],
    collection: &str,
    exclude_id: &uuid::Uuid,
) -> Result<Vec<SimilarConversation>, Box<dyn std::error::Error>> {
    
    let search_params = crate::vector::SearchParams::new(query_embedding.to_vec(), 5)
        .with_score_threshold(0.6);
    
    let search_result = vector_ops.search_vectors(search_params, Some(collection)).await?;
    
    let mut similar = Vec::new();
    for result in search_result.results {
        if result.id != *exclude_id {
            similar.push(SimilarConversation {
                id: result.id.to_string(),
                prompt: result.get_metadata::<String>("prompt").unwrap_or_default(),
                response_summary: result.get_metadata::<String>("response")
                    .unwrap_or_default()
                    .chars()
                    .take(100)
                    .collect(),
                similarity_score: result.score as f64,
                timestamp: result.get_metadata::<String>("timestamp").unwrap_or_default(),
            });
        }
    }
    
    Ok(similar)
}

// ================================================================================================
// EXAMPLE USAGE
// ================================================================================================

/*
EXAMPLE API CALL:

curl -X POST http://localhost:3000/api/v1/generate/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I learn machine learning?",
    "user_id": "user_123",
    "session_id": "session_456", 
    "store_conversation": true,
    "use_context": true,
    "context_limit": 3,
    "max_tokens": 150
  }'

EXAMPLE RESPONSE:

{
  "text": "Based on your previous questions about programming, I recommend starting with Python for machine learning...",
  "request_id": "req_789",
  "conversation_id": "conv_101112",
  "used_context": [
    {
      "conversation_id": "conv_101110",
      "summary": "Q: What programming language should I start with? A: Python is great for beginners...",
      "relevance_score": 0.85,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "embedding_stored": true,
  "similar_conversations": [
    {
      "id": "conv_101109",
      "prompt": "Best resources for learning AI",
      "response_summary": "I recommend starting with online courses like...",
      "similarity_score": 0.78,
      "timestamp": "2024-01-14T15:20:00Z"
    }
  ],
  "personalization_applied": true,
  "processing_time_ms": 450,
  "tokens_generated": 87,
  "tokens_per_second": 12.5
}
*/