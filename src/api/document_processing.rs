// ================================================================================================
// DAY 10: DOCUMENT PROCESSING API ENDPOINTS
// ================================================================================================
//
// RESTful API endpoints for:
// - Document ingestion and parsing
// - Intelligent chunking
// - Incremental updates and deduplication
// - Document version management
//
// ================================================================================================

use axum::{
    extract::{Path, State, Multipart},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use tracing::info;

use crate::vector::{
    DocumentIngestionPipeline, IntelligentChunker, IncrementalUpdateManager,
    RawDocument, ProcessedDocument,
    DocumentFormat, ChunkingStrategy,
    create_document_pipeline, create_semantic_chunker, create_incremental_manager,
};
use crate::errors::AppError;

// ================================================================================================
// API STATE
// ================================================================================================

#[derive(Clone)]
pub struct DocumentProcessingApiState {
    pub ingestion_pipeline: std::sync::Arc<tokio::sync::Mutex<DocumentIngestionPipeline>>,
    pub chunker: std::sync::Arc<IntelligentChunker>,
    pub update_manager: std::sync::Arc<tokio::sync::Mutex<IncrementalUpdateManager>>,
}

impl DocumentProcessingApiState {
    pub fn new() -> Self {
        Self {
            ingestion_pipeline: std::sync::Arc::new(tokio::sync::Mutex::new(create_document_pipeline())),
            chunker: std::sync::Arc::new(create_semantic_chunker(500)),
            update_manager: std::sync::Arc::new(tokio::sync::Mutex::new(create_incremental_manager())),
        }
    }
}

// ================================================================================================
// REQUEST/RESPONSE TYPES
// ================================================================================================

// INGESTION REQUESTS
#[derive(Debug, Deserialize)]
pub struct IngestDocumentRequest {
    pub content: String,
    pub format: Option<DocumentFormat>,
    pub source_path: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct BatchIngestRequest {
    pub file_paths: Vec<String>,
    pub batch_size: Option<usize>,
    pub parallel_processing: Option<bool>,
}

// CHUNKING REQUESTS
#[derive(Debug, Deserialize)]
pub struct ChunkExistingDocumentRequest {
    pub strategy: Option<ChunkingStrategy>,
    pub config_overrides: Option<ChunkingConfigOverrides>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkContentRequest {
    pub content: String,
    pub strategy: Option<ChunkingStrategy>,
    pub config_overrides: Option<ChunkingConfigOverrides>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkingConfigOverrides {
    pub target_size: Option<usize>,
    pub preserve_metadata: Option<bool>,
    pub add_context_headers: Option<bool>,
}

// UPDATE REQUESTS
#[derive(Debug, Deserialize)]
pub struct UpdateDocumentRequest {
    pub document_id: Uuid,
    pub new_content: String,
    pub chunk_ids: Vec<Uuid>,
    pub force_update: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct DeduplicationRequest {
    pub similarity_threshold: Option<f32>,
    pub strategy: Option<String>,
}

// RESPONSES
#[derive(Debug, Serialize)]
pub struct IngestDocumentResponse {
    pub document_id: Uuid,
    pub original_id: Uuid,
    pub sections_count: usize,
    pub total_tokens: usize,
    pub processing_time_ms: u64,
    pub format: DocumentFormat,
}

#[derive(Debug, Serialize)]
pub struct BatchIngestResponse {
    pub total_processed: usize,
    pub successful: usize,
    pub failed: usize,
    pub processing_time_ms: u64,
    pub results: Vec<IngestResult>,
}

#[derive(Debug, Serialize)]
pub struct IngestResult {
    pub file_path: String,
    pub success: bool,
    pub document_id: Option<Uuid>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDocumentResponse {
    pub document_id: Uuid,
    pub chunks: Vec<ChunkInfo>,
    pub total_chunks: usize,
    pub total_tokens: usize,
    pub average_chunk_size: f32,
    pub processing_time_ms: u64,
    pub quality_metrics: ChunkQualityInfo,
}

#[derive(Debug, Serialize)]
pub struct ChunkInfo {
    pub id: Uuid,
    pub content: String,
    pub token_count: usize,
    pub chunk_index: usize,
    pub chunk_type: String,
    pub has_overlap: bool,
}

#[derive(Debug, Serialize)]
pub struct ChunkQualityInfo {
    pub boundary_preservation_score: f32,
    pub size_consistency_score: f32,
    pub overlap_coverage_score: f32,
    pub context_preservation_score: f32,
}

#[derive(Debug, Serialize)]
pub struct UpdateDocumentResponse {
    pub document_id: Uuid,
    pub old_version: Option<Uuid>,
    pub new_version: Uuid,
    pub change_type: String,
    pub chunks_updated: Vec<Uuid>,
    pub chunks_added: Vec<Uuid>,
    pub chunks_removed: Vec<Uuid>,
    pub deduplication_applied: usize,
    pub processing_time_ms: u64,
    pub storage_savings: StorageSavingsInfo,
}

#[derive(Debug, Serialize)]
pub struct StorageSavingsInfo {
    pub size_change_bytes: i64,
    pub vector_count_change: i32,
    pub efficiency_improvement: f32,
}

#[derive(Debug, Serialize)]
pub struct DeduplicationResponse {
    pub candidates_found: usize,
    pub duplicates_processed: usize,
    pub storage_saved_bytes: usize,
    pub vectors_eliminated: usize,
    pub efficiency_gain: f32,
}

#[derive(Debug, Serialize)]
pub struct DocumentStatsResponse {
    pub total_documents: usize,
    pub total_versions: usize,
    pub total_chunks: usize,
    pub unique_content_hashes: usize,
    pub duplicate_groups: usize,
    pub average_document_size: f32,
    pub storage_efficiency: f32,
}

// ================================================================================================
// ROUTER SETUP
// ================================================================================================

pub fn create_document_processing_router() -> Router<DocumentProcessingApiState> {
    Router::new()
        // Document Ingestion
        .route("/api/v1/documents/ingest", post(ingest_document))
        .route("/api/v1/documents/upload", post(upload_file_document))
        .route("/api/v1/documents/ingest/batch", post(batch_ingest))
        .route("/api/v1/documents/{id}", get(get_document))
        
        // Document Chunking
        .route("/api/v1/documents/{id}/chunk", post(chunk_existing_document))
        .route("/api/v1/documents/chunk", post(chunk_content))
        .route("/api/v1/documents/{id}/chunks", get(get_document_chunks))
        
        // Document Updates
        .route("/api/v1/documents/update", post(update_document))
        .route("/api/v1/documents/{id}/versions", get(get_document_versions))
        
        // Deduplication
        .route("/api/v1/documents/deduplicate", post(run_deduplication))
        .route("/api/v1/documents/duplicates", get(find_duplicates))
        
        // Statistics and Monitoring
        .route("/api/v1/documents/stats", get(get_document_stats))
        .route("/api/v1/documents/{id}/stats", get(get_document_specific_stats))
}

// ================================================================================================
// DOCUMENT INGESTION ENDPOINTS
// ================================================================================================

/// Ingest a single document
#[axum::debug_handler]
pub async fn ingest_document(
    State(state): State<DocumentProcessingApiState>,
    Json(request): Json<IngestDocumentRequest>,
) -> Result<Json<IngestDocumentResponse>, AppError> {
    info!("📄 Ingesting single document");
    
    // Create raw document
    let format = request.format.unwrap_or(DocumentFormat::PlainText);
    let mut raw_doc = RawDocument::new(request.content, format.clone());
    
    if let Some(path) = request.source_path {
        raw_doc.source_path = Some(path);
    }
    
    if let Some(metadata) = request.metadata {
        raw_doc.metadata = metadata;
    }
    
    // Process document
    let processed = {
        let mut pipeline = state.ingestion_pipeline.lock().await;
        pipeline.process_document(raw_doc.clone()).await
            .map_err(|e| AppError::Processing(format!("Document ingestion failed: {}", e)))?
    };
    
    let response = IngestDocumentResponse {
        document_id: processed.id,
        original_id: processed.original_id,
        sections_count: processed.sections.len(),
        total_tokens: processed.total_tokens,
        processing_time_ms: 0, // Would track in real implementation
        format,
    };
    
    info!("✅ Document ingested: {} sections, {} tokens", 
          response.sections_count, response.total_tokens);
    
    Ok(Json(response))
}

/// Upload and ingest a document from multipart file
#[axum::debug_handler]
pub async fn upload_file_document(
    State(state): State<DocumentProcessingApiState>,
    mut multipart: Multipart,
) -> Result<Json<IngestDocumentResponse>, AppError> {
    info!("📁 Processing multipart file upload");

    let mut file_content: Option<String> = None;
    let mut file_name: Option<String> = None;
    let mut metadata: HashMap<String, String> = HashMap::new();

    // Process multipart fields with timeout protection
    let multipart_timeout = tokio::time::Duration::from_secs(30);
    let multipart_future = async {
        while let Some(field) = multipart.next_field().await
            .map_err(|e| AppError::Processing(format!("Multipart parsing failed: {}", e)))? {
        
        let field_name = field.name().unwrap_or("unknown").to_string();
        
        match field_name.as_str() {
            "file" => {
                // Extract filename from field
                if let Some(filename) = field.file_name() {
                    file_name = Some(filename.to_string());
                }
                
                // Read file content
                let data = field.bytes().await
                    .map_err(|e| AppError::Processing(format!("Failed to read file data: {}", e)))?;
                
                file_content = Some(String::from_utf8_lossy(&data).to_string());
                info!("📄 File uploaded: {} ({} bytes)", file_name.as_ref().unwrap_or(&"unknown".to_string()), data.len());
            },
            "author" => {
                let value = field.text().await
                    .map_err(|e| AppError::Processing(format!("Failed to read author field: {}", e)))?;
                metadata.insert("author".to_string(), value);
            },
            "category" => {
                let value = field.text().await
                    .map_err(|e| AppError::Processing(format!("Failed to read category field: {}", e)))?;
                metadata.insert("category".to_string(), value);
            },
            "tags" => {
                let value = field.text().await
                    .map_err(|e| AppError::Processing(format!("Failed to read tags field: {}", e)))?;
                metadata.insert("tags".to_string(), value);
            },
            _ => {
                // Handle other metadata fields
                let value = field.text().await
                    .map_err(|e| AppError::Processing(format!("Failed to read field {}: {}", field_name, e)))?;
                metadata.insert(field_name, value);
            }
        }
        }
        Ok::<(), AppError>(())
    };
    
    // Execute multipart processing with timeout
    tokio::time::timeout(multipart_timeout, multipart_future)
        .await
        .map_err(|_| AppError::Processing("Multipart processing timeout (30s)".to_string()))?
        .map_err(|e| e)?;

    // Validate required fields
    let content = file_content.ok_or_else(|| {
        AppError::Validation("No file provided in multipart request".to_string())
    })?;

    let filename = file_name.unwrap_or_else(|| "uploaded_file".to_string());

    // Detect format from file extension
    let format = detect_format_from_filename(&filename);
    
    // Create raw document
    let mut raw_doc = RawDocument::new(content, format.clone());
    raw_doc.source_path = Some(filename.clone());
    raw_doc.metadata = metadata;

    // Process document
    let processed = {
        let mut pipeline = state.ingestion_pipeline.lock().await;
        pipeline.process_document(raw_doc.clone()).await
            .map_err(|e| AppError::Processing(format!("Document ingestion failed: {}", e)))?
    };

    let response = IngestDocumentResponse {
        document_id: processed.id,
        original_id: processed.original_id,
        sections_count: processed.sections.len(),
        total_tokens: processed.total_tokens,
        processing_time_ms: 0,
        format,
    };

    info!("✅ File document ingested: {} - {} sections, {} tokens", 
          filename, response.sections_count, response.total_tokens);

    Ok(Json(response))
}

/// Batch ingest multiple documents
async fn batch_ingest(
    State(state): State<DocumentProcessingApiState>,
    Json(request): Json<BatchIngestRequest>,
) -> Result<Json<BatchIngestResponse>, AppError> {
    info!("📚 Starting batch ingestion of {} files", request.file_paths.len());
    
    let start_time = std::time::Instant::now();
    let results = {
        let mut pipeline = state.ingestion_pipeline.lock().await;
        pipeline.ingest_batch(request.file_paths.clone()).await
    };
    
    let mut ingest_results = Vec::new();
    let mut successful = 0;
    let mut failed = 0;
    
    for (file_path, result) in request.file_paths.iter().zip(results.iter()) {
        match result {
            Ok(doc) => {
                successful += 1;
                ingest_results.push(IngestResult {
                    file_path: file_path.clone(),
                    success: true,
                    document_id: Some(doc.id),
                    error: None,
                });
            },
            Err(e) => {
                failed += 1;
                ingest_results.push(IngestResult {
                    file_path: file_path.clone(),
                    success: false,
                    document_id: None,
                    error: Some(e.to_string()),
                });
            }
        }
    }
    
    let response = BatchIngestResponse {
        total_processed: results.len(),
        successful,
        failed,
        processing_time_ms: start_time.elapsed().as_millis() as u64,
        results: ingest_results,
    };
    
    info!("✅ Batch ingestion complete: {}/{} successful", successful, results.len());
    Ok(Json(response))
}

/// Get document by ID
async fn get_document(
    State(state): State<DocumentProcessingApiState>,
    Path(document_id): Path<Uuid>,
) -> Result<Json<ProcessedDocument>, AppError> {
    info!("📋 Retrieving document: {}", document_id);
    
    let pipeline = state.ingestion_pipeline.lock().await;
    let document = pipeline.get_document(&document_id)
        .ok_or_else(|| AppError::Processing(format!("Document not found: {}", document_id)))?;
    
    info!("✅ Document retrieved: {} sections", document.sections.len());
    Ok(Json(document.clone()))
}

// ================================================================================================
// DOCUMENT CHUNKING ENDPOINTS
// ================================================================================================

/// Chunk existing document by ID - retrieves content from storage
async fn chunk_existing_document(
    State(state): State<DocumentProcessingApiState>,
    Path(document_id): Path<Uuid>,
    Json(_request): Json<ChunkExistingDocumentRequest>,
) -> Result<Json<ChunkDocumentResponse>, AppError> {
    info!("🔪 Chunking existing document: {}", document_id);
    
    // Retrieve document content from storage
    let content = {
        let pipeline = state.ingestion_pipeline.lock().await;
        pipeline.get_document_content(&document_id)
            .ok_or_else(|| AppError::Processing(format!("Document not found: {}", document_id)))?
    };
    
    let chunking_result = state.chunker.chunk_document(document_id, &content)
        .map_err(|e| AppError::Processing(format!("Document chunking failed: {}", e)))?;
    
    let chunks: Vec<ChunkInfo> = chunking_result.chunks.iter().map(|chunk| {
        ChunkInfo {
            id: chunk.id,
            content: chunk.content.clone(),
            token_count: chunk.token_count,
            chunk_index: chunk.chunk_index,
            chunk_type: format!("{:?}", chunk.chunk_type),
            has_overlap: chunk.overlap_info.is_some(),
        }
    }).collect();
    
    let response = ChunkDocumentResponse {
        document_id,
        total_chunks: chunking_result.total_chunks,
        total_tokens: chunking_result.total_tokens,
        average_chunk_size: chunking_result.average_chunk_size,
        processing_time_ms: chunking_result.processing_time_ms,
        quality_metrics: ChunkQualityInfo {
            boundary_preservation_score: chunking_result.quality_metrics.boundary_preservation_score,
            size_consistency_score: chunking_result.quality_metrics.size_consistency_score,
            overlap_coverage_score: chunking_result.quality_metrics.overlap_coverage_score,
            context_preservation_score: chunking_result.quality_metrics.context_preservation_score,
        },
        chunks,
    };
    
    info!("✅ Document chunked: {} chunks generated", response.total_chunks);
    Ok(Json(response))
}

/// Chunk arbitrary content - doesn't require existing document
async fn chunk_content(
    State(state): State<DocumentProcessingApiState>,
    Json(request): Json<ChunkContentRequest>,
) -> Result<Json<ChunkDocumentResponse>, AppError> {
    let temp_id = Uuid::new_v4();
    info!("🔪 Chunking content (temporary ID: {})", temp_id);
    
    let chunking_result = state.chunker.chunk_document(temp_id, &request.content)
        .map_err(|e| AppError::Processing(format!("Content chunking failed: {}", e)))?;
    
    let chunks: Vec<ChunkInfo> = chunking_result.chunks.iter().map(|chunk| {
        ChunkInfo {
            id: chunk.id,
            content: chunk.content.clone(),
            token_count: chunk.token_count,
            chunk_index: chunk.chunk_index,
            chunk_type: format!("{:?}", chunk.chunk_type),
            has_overlap: chunk.overlap_info.is_some(),
        }
    }).collect();
    
    let response = ChunkDocumentResponse {
        document_id: temp_id,
        total_chunks: chunking_result.total_chunks,
        total_tokens: chunking_result.total_tokens,
        average_chunk_size: chunking_result.average_chunk_size,
        processing_time_ms: chunking_result.processing_time_ms,
        quality_metrics: ChunkQualityInfo {
            boundary_preservation_score: chunking_result.quality_metrics.boundary_preservation_score,
            size_consistency_score: chunking_result.quality_metrics.size_consistency_score,
            overlap_coverage_score: chunking_result.quality_metrics.overlap_coverage_score,
            context_preservation_score: chunking_result.quality_metrics.context_preservation_score,
        },
        chunks,
    };
    
    info!("✅ Content chunked: {} chunks generated", response.total_chunks);
    Ok(Json(response))
}

/// Get chunks for a specific document
async fn get_document_chunks(
    State(_state): State<DocumentProcessingApiState>,
    Path(document_id): Path<Uuid>,
) -> Result<Json<Vec<ChunkInfo>>, AppError> {
    info!("📋 Retrieving chunks for document: {}", document_id);
    
    // In a real implementation, this would query the vector database
    // For now, return placeholder data
    let chunks = vec![
        ChunkInfo {
            id: Uuid::new_v4(),
            content: "Sample chunk content".to_string(),
            token_count: 25,
            chunk_index: 0,
            chunk_type: "Content".to_string(),
            has_overlap: false,
        }
    ];
    
    Ok(Json(chunks))
}

// ================================================================================================
// DOCUMENT UPDATE ENDPOINTS
// ================================================================================================

/// Update a document with incremental changes
async fn update_document(
    State(state): State<DocumentProcessingApiState>,
    Json(request): Json<UpdateDocumentRequest>,
) -> Result<Json<UpdateDocumentResponse>, AppError> {
    info!("🔄 Updating document: {}", request.document_id);
    
    let update_result = {
        let mut manager = state.update_manager.lock().await;
        manager.update_document(
            request.document_id,
            &request.new_content,
            request.chunk_ids,
        ).await.map_err(|e| AppError::Processing(format!("Document update failed: {}", e)))?
    };
    
    let response = UpdateDocumentResponse {
        document_id: update_result.document_id,
        old_version: update_result.old_version,
        new_version: update_result.new_version,
        change_type: format!("{:?}", update_result.change_summary.change_type),
        chunks_updated: update_result.chunks_updated,
        chunks_added: update_result.chunks_added,
        chunks_removed: update_result.chunks_removed,
        deduplication_applied: update_result.deduplication_applied.len(),
        processing_time_ms: update_result.processing_time_ms,
        storage_savings: StorageSavingsInfo {
            size_change_bytes: update_result.storage_impact.size_change_bytes,
            vector_count_change: update_result.storage_impact.vector_count_change,
            efficiency_improvement: update_result.storage_impact.efficiency_improvement,
        },
    };
    
    info!("✅ Document updated: {} -> {}", 
          response.old_version.map(|id| id.to_string()).unwrap_or("new".to_string()),
          response.new_version);
    
    Ok(Json(response))
}

/// Get version history for a document
async fn get_document_versions(
    State(state): State<DocumentProcessingApiState>,
    Path(document_id): Path<Uuid>,
) -> Result<Json<Vec<HashMap<String, serde_json::Value>>>, AppError> {
    info!("📚 Retrieving version history for: {}", document_id);
    
    let versions = {
        let manager = state.update_manager.lock().await;
        manager.get_version_history(document_id).cloned()
    };
    
    match versions {
        Some(version_list) => {
            let version_info: Vec<_> = version_list.iter().map(|v| {
                let mut info = HashMap::new();
                info.insert("id".to_string(), serde_json::json!(v.id));
                info.insert("version_number".to_string(), serde_json::json!(v.version_number));
                info.insert("created_at".to_string(), serde_json::json!(v.created_at));
                info.insert("change_type".to_string(), serde_json::json!(v.change_summary.change_type));
                info.insert("sections_added".to_string(), serde_json::json!(v.change_summary.sections_added));
                info.insert("sections_modified".to_string(), serde_json::json!(v.change_summary.sections_modified));
                info.insert("sections_removed".to_string(), serde_json::json!(v.change_summary.sections_removed));
                info.insert("similarity_score".to_string(), serde_json::json!(v.change_summary.similarity_to_previous));
                info
            }).collect();
            
            Ok(Json(version_info))
        },
        None => Ok(Json(Vec::new()))
    }
}

// ================================================================================================
// DEDUPLICATION ENDPOINTS
// ================================================================================================

/// Run deduplication analysis
async fn run_deduplication(
    State(state): State<DocumentProcessingApiState>,
    Json(_request): Json<DeduplicationRequest>,
) -> Result<Json<DeduplicationResponse>, AppError> {
    info!("🔍 Running deduplication analysis");
    
    let (candidates, processed_chunks) = {
        let mut manager = state.update_manager.lock().await;
        let candidates = manager.find_duplicates_global().await
            .map_err(|e| AppError::Processing(format!("Deduplication failed: {}", e)))?;
        
        let processed_chunks = manager.apply_deduplication(candidates.clone()).await
            .map_err(|e| AppError::Processing(format!("Deduplication application failed: {}", e)))?;
        
        (candidates, processed_chunks)
    };
    
    let total_savings: usize = candidates.iter()
        .map(|c| c.potential_savings.bytes_saved)
        .sum();
    
    let vectors_eliminated: usize = candidates.iter()
        .map(|c| c.potential_savings.vectors_eliminated)
        .sum();
    
    let response = DeduplicationResponse {
        candidates_found: candidates.len(),
        duplicates_processed: processed_chunks.len(),
        storage_saved_bytes: total_savings,
        vectors_eliminated,
        efficiency_gain: if total_savings > 0 { 0.15 } else { 0.0 }, // Placeholder
    };
    
    info!("✅ Deduplication complete: {} duplicates processed, {} bytes saved", 
          response.duplicates_processed, response.storage_saved_bytes);
    
    Ok(Json(response))
}

/// Find duplicate candidates
#[axum::debug_handler]  
pub async fn find_duplicates(
    State(state): State<DocumentProcessingApiState>,
) -> Result<Json<Vec<HashMap<String, serde_json::Value>>>, AppError> {
    info!("🔍 Finding duplicate candidates");
    
    let candidates = {
        let mut manager = state.update_manager.lock().await;
        manager.find_duplicates_global().await
            .map_err(|e| AppError::Processing(format!("Duplicate detection failed: {}", e)))?
    };
    
    let candidate_info: Vec<_> = candidates.iter().map(|c| {
        let mut info = HashMap::new();
        info.insert("primary_chunk_id".to_string(), serde_json::json!(c.primary_chunk_id));
        info.insert("duplicate_count".to_string(), serde_json::json!(c.duplicate_chunk_ids.len()));
        info.insert("similarity_score".to_string(), serde_json::json!(c.similarity_score));
        info.insert("content_overlap".to_string(), serde_json::json!(c.content_overlap));
        info.insert("strategy".to_string(), serde_json::json!(c.dedup_strategy));
        info.insert("bytes_saved".to_string(), serde_json::json!(c.potential_savings.bytes_saved));
        info
    }).collect();
    
    Ok(Json(candidate_info))
}

// ================================================================================================
// STATISTICS ENDPOINTS
// ================================================================================================

/// Get overall document processing statistics
async fn get_document_stats(
    State(state): State<DocumentProcessingApiState>,
) -> Result<Json<DocumentStatsResponse>, AppError> {
    info!("📊 Retrieving document processing statistics");
    
    let (ingestion_stats, update_stats) = {
        let pipeline = state.ingestion_pipeline.lock().await;
        let manager = state.update_manager.lock().await;
        (pipeline.get_stats().clone(), manager.get_stats())
    };
    
    let response = DocumentStatsResponse {
        total_documents: update_stats.total_documents,
        total_versions: update_stats.total_versions,
        total_chunks: ingestion_stats.total_sections,
        unique_content_hashes: update_stats.unique_content_hashes,
        duplicate_groups: update_stats.duplicate_content_groups,
        average_document_size: if update_stats.total_documents > 0 {
            ingestion_stats.total_tokens as f32 / update_stats.total_documents as f32
        } else { 0.0 },
        storage_efficiency: update_stats.cache_hit_rate,
    };
    
    Ok(Json(response))
}

/// Get statistics for a specific document
async fn get_document_specific_stats(
    State(_state): State<DocumentProcessingApiState>,
    Path(document_id): Path<Uuid>,
) -> Result<Json<HashMap<String, serde_json::Value>>, AppError> {
    info!("📊 Retrieving statistics for document: {}", document_id);
    
    // Placeholder implementation
    let mut stats = HashMap::new();
    stats.insert("document_id".to_string(), serde_json::json!(document_id));
    stats.insert("version_count".to_string(), serde_json::json!(1));
    stats.insert("total_chunks".to_string(), serde_json::json!(5));
    stats.insert("total_tokens".to_string(), serde_json::json!(1250));
    stats.insert("last_updated".to_string(), serde_json::json!("2024-01-01T00:00:00Z"));
    
    Ok(Json(stats))
}

// ================================================================================================
// HELPER FUNCTIONS
// ================================================================================================

/// Detect document format from filename extension
fn detect_format_from_filename(filename: &str) -> DocumentFormat {
    let extension = filename.split('.').last().unwrap_or("").to_lowercase();
    match extension.as_str() {
        "md" | "markdown" => DocumentFormat::Markdown,
        "json" => DocumentFormat::Json,
        "csv" => DocumentFormat::Csv,
        "txt" | "text" => DocumentFormat::PlainText,
        _ => DocumentFormat::PlainText, // Default to plain text for unknown extensions
    }
}