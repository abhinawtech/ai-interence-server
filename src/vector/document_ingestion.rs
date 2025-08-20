// ================================================================================================
// DAY 10.1: DOCUMENT INGESTION AND PARSING PIPELINE
// ================================================================================================
//
// Intelligent document processing system that can:
// - Parse multiple document formats (text, markdown, JSON, CSV)
// - Extract metadata and structure
// - Handle large documents efficiently
// - Provide progress tracking for batch operations
//
// ================================================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tokio::fs;
use tracing::{info, warn, error};

// ================================================================================================
// CORE TYPES
// ================================================================================================

/// Document format types supported by the ingestion pipeline
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DocumentFormat {
    PlainText,
    Markdown,
    Json,
    Csv,
    Unknown,
}

impl DocumentFormat {
    pub fn from_extension(path: &str) -> Self {
        let extension = PathBuf::from(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();
            
        match extension.as_str() {
            "txt" => Self::PlainText,
            "md" | "markdown" => Self::Markdown,
            "json" => Self::Json,
            "csv" => Self::Csv,
            _ => Self::Unknown,
        }
    }
}

/// Raw document before processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDocument {
    pub id: Uuid,
    pub source_path: Option<String>,
    pub content: String,
    pub format: DocumentFormat,
    pub size_bytes: usize,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl RawDocument {
    pub fn new(content: String, format: DocumentFormat) -> Self {
        let size_bytes = content.len();
        Self {
            id: Uuid::new_v4(),
            source_path: None,
            content,
            format,
            size_bytes,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn from_file(path: String, content: String) -> Self {
        let format = DocumentFormat::from_extension(&path);
        let size_bytes = content.len();
        
        Self {
            id: Uuid::new_v4(),
            source_path: Some(path.clone()),
            content,
            format,
            size_bytes,
            created_at: Utc::now(),
            metadata: HashMap::from([
                ("source".to_string(), path),
                ("ingested_at".to_string(), Utc::now().to_rfc3339()),
            ]),
        }
    }
}

/// Processed document structure with extracted sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDocument {
    pub id: Uuid,
    pub original_id: Uuid,
    pub title: Option<String>,
    pub sections: Vec<DocumentSection>,
    pub format: DocumentFormat,
    pub total_tokens: usize,
    pub processed_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// A section/chunk of a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSection {
    pub id: Uuid,
    pub document_id: Uuid,
    pub content: String,
    pub section_type: SectionType,
    pub order_index: usize,
    pub token_count: usize,
    pub metadata: HashMap<String, String>,
}

/// Types of document sections
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SectionType {
    Title,
    Header,
    Paragraph,
    Code,
    List,
    Table,
    Metadata,
}

/// Ingestion job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    pub batch_size: usize,
    pub max_section_tokens: usize,
    pub extract_metadata: bool,
    pub preserve_structure: bool,
    pub parallel_processing: bool,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            max_section_tokens: 500,
            extract_metadata: true,
            preserve_structure: true,
            parallel_processing: true,
        }
    }
}

/// Processing results and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionStats {
    pub documents_processed: usize,
    pub total_sections: usize,
    pub total_tokens: usize,
    pub processing_time_ms: u64,
    pub errors: Vec<IngestionError>,
    pub format_breakdown: HashMap<DocumentFormat, usize>,
}

impl Default for IngestionStats {
    fn default() -> Self {
        Self {
            documents_processed: 0,
            total_sections: 0,
            total_tokens: 0,
            processing_time_ms: 0,
            errors: Vec::new(),
            format_breakdown: HashMap::new(),
        }
    }
}

// ================================================================================================
// ERROR HANDLING
// ================================================================================================

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum IngestionError {
    #[error("File reading error: {0}")]
    FileRead(String),
    
    #[error("Format parsing error: {0}")]
    FormatParsing(String),
    
    #[error("Content processing error: {0}")]
    ContentProcessing(String),
    
    #[error("Token limit exceeded: {current} > {max}")]
    TokenLimitExceeded { current: usize, max: usize },
    
    #[error("Invalid document format: {0}")]
    InvalidFormat(String),
}

pub type IngestionResult<T> = Result<T, IngestionError>;

// ================================================================================================
// DOCUMENT INGESTION PIPELINE
// ================================================================================================

/// Main document ingestion pipeline
pub struct DocumentIngestionPipeline {
    config: IngestionConfig,
    stats: IngestionStats,
    document_store: std::collections::HashMap<Uuid, ProcessedDocument>,
}

impl DocumentIngestionPipeline {
    pub fn new(config: IngestionConfig) -> Self {
        Self {
            config,
            stats: IngestionStats::default(),
            document_store: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_default_config() -> Self {
        Self::new(IngestionConfig::default())
    }
    
    /// Process a single document from file path
    pub async fn ingest_file(&mut self, file_path: &str) -> IngestionResult<ProcessedDocument> {
        info!("📄 Ingesting document from file: {}", file_path);
        
        // Read file content
        let content = fs::read_to_string(file_path)
            .await
            .map_err(|e| IngestionError::FileRead(format!("Failed to read {}: {}", file_path, e)))?;
        
        let raw_doc = RawDocument::from_file(file_path.to_string(), content);
        self.process_document(raw_doc).await
    }
    
    /// Process multiple files in batch
    pub async fn ingest_batch(&mut self, file_paths: Vec<String>) -> Vec<IngestionResult<ProcessedDocument>> {
        info!("📚 Starting batch ingestion of {} documents", file_paths.len());
        let start_time = std::time::Instant::now();
        
        let mut results = Vec::new();
        
        // For now, always use sequential processing to avoid borrow checker issues
        // In a production system, we'd use a different approach for parallel processing
        for file_path in file_paths {
            let result = self.ingest_file(&file_path).await;
            results.push(result);
        }
        
        // Update stats
        self.stats.processing_time_ms = start_time.elapsed().as_millis() as u64;
        self.stats.documents_processed = results.iter().filter(|r| r.is_ok()).count();
        
        info!("✅ Batch ingestion complete: {} documents processed in {}ms", 
              self.stats.documents_processed, self.stats.processing_time_ms);
        
        results
    }
    
    /// Process a raw document into structured format
    pub async fn process_document(&mut self, raw_doc: RawDocument) -> IngestionResult<ProcessedDocument> {
        info!("🔄 Processing document: {} ({:?})", raw_doc.id, raw_doc.format);
        
        // Parse content based on format
        let sections = match raw_doc.format {
            DocumentFormat::PlainText => self.parse_plain_text(&raw_doc.content)?,
            DocumentFormat::Markdown => self.parse_markdown(&raw_doc.content)?,
            DocumentFormat::Json => self.parse_json(&raw_doc.content)?,
            DocumentFormat::Csv => self.parse_csv(&raw_doc.content)?,
            DocumentFormat::Unknown => {
                warn!("Unknown document format, treating as plain text");
                self.parse_plain_text(&raw_doc.content)?
            }
        };
        
        // Calculate total tokens
        let total_tokens: usize = sections.iter().map(|s| s.token_count).sum();
        
        // Extract title
        let title = self.extract_title(&sections);
        
        let processed = ProcessedDocument {
            id: Uuid::new_v4(),
            original_id: raw_doc.id,
            title,
            sections,
            format: raw_doc.format.clone(),
            total_tokens,
            processed_at: Utc::now(),
            metadata: raw_doc.metadata,
        };
        
        // Update stats
        self.stats.total_sections += processed.sections.len();
        self.stats.total_tokens += total_tokens;
        *self.stats.format_breakdown.entry(raw_doc.format).or_insert(0) += 1;
        
        // Store the processed document
        self.document_store.insert(processed.id, processed.clone());
        
        info!("✅ Document processed: {} sections, {} tokens", 
              processed.sections.len(), total_tokens);
        
        Ok(processed)
    }
    
    /// Retrieve a document by ID
    pub fn get_document(&self, document_id: &Uuid) -> Option<&ProcessedDocument> {
        self.document_store.get(document_id)
    }
    
    /// Get the full text content of a document
    pub fn get_document_content(&self, document_id: &Uuid) -> Option<String> {
        self.document_store.get(document_id).map(|doc| {
            doc.sections.iter()
                .map(|section| section.content.clone())
                .collect::<Vec<String>>()
                .join("\n\n")
        })
    }
    
    /// Get all document IDs
    pub fn get_document_ids(&self) -> Vec<Uuid> {
        self.document_store.keys().cloned().collect()
    }
    
    pub fn get_stats(&self) -> &IngestionStats {
        &self.stats
    }
    
    pub fn reset_stats(&mut self) {
        self.stats = IngestionStats::default();
    }
}

// ================================================================================================
// FORMAT PARSERS
// ================================================================================================

impl DocumentIngestionPipeline {
    /// Parse plain text into logical sections
    fn parse_plain_text(&self, content: &str) -> IngestionResult<Vec<DocumentSection>> {
        let mut sections = Vec::new();
        let paragraphs: Vec<&str> = content.split("\n\n").filter(|p| !p.trim().is_empty()).collect();
        
        for (index, paragraph) in paragraphs.iter().enumerate() {
            let token_count = estimate_tokens(paragraph);
            
            // Split large paragraphs if needed
            if token_count > self.config.max_section_tokens {
                let chunks = self.split_large_section(paragraph, self.config.max_section_tokens);
                for (chunk_idx, chunk) in chunks.iter().enumerate() {
                    sections.push(DocumentSection {
                        id: Uuid::new_v4(),
                        document_id: Uuid::new_v4(), // Will be set later
                        content: chunk.to_string(),
                        section_type: SectionType::Paragraph,
                        order_index: index * 1000 + chunk_idx,
                        token_count: estimate_tokens(chunk),
                        metadata: HashMap::from([
                            ("chunk_of".to_string(), index.to_string()),
                        ]),
                    });
                }
            } else {
                sections.push(DocumentSection {
                    id: Uuid::new_v4(),
                    document_id: Uuid::new_v4(),
                    content: paragraph.to_string(),
                    section_type: SectionType::Paragraph,
                    order_index: index,
                    token_count,
                    metadata: HashMap::new(),
                });
            }
        }
        
        Ok(sections)
    }
    
    /// Parse Markdown with structure preservation
    fn parse_markdown(&self, content: &str) -> IngestionResult<Vec<DocumentSection>> {
        let mut sections = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut current_section = String::new();
        let mut section_type = SectionType::Paragraph;
        let mut order_index = 0;
        
        for line in lines {
            if line.starts_with('#') {
                // Save previous section if not empty
                if !current_section.trim().is_empty() {
                    sections.push(DocumentSection {
                        id: Uuid::new_v4(),
                        document_id: Uuid::new_v4(),
                        content: current_section.trim().to_string(),
                        section_type: section_type.clone(),
                        order_index,
                        token_count: estimate_tokens(&current_section),
                        metadata: HashMap::new(),
                    });
                    order_index += 1;
                    current_section.clear();
                }
                
                // Start new header section
                section_type = SectionType::Header;
                current_section = line.to_string();
            } else if line.starts_with("```") {
                // Code block detection
                if section_type == SectionType::Code {
                    current_section.push_str(line);
                    current_section.push('\n');
                    // End code block
                    sections.push(DocumentSection {
                        id: Uuid::new_v4(),
                        document_id: Uuid::new_v4(),
                        content: current_section.trim().to_string(),
                        section_type: SectionType::Code,
                        order_index,
                        token_count: estimate_tokens(&current_section),
                        metadata: HashMap::from([
                            ("language".to_string(), extract_language_from_code_fence(line)),
                        ]),
                    });
                    order_index += 1;
                    current_section.clear();
                    section_type = SectionType::Paragraph;
                } else {
                    // Start code block
                    if !current_section.trim().is_empty() {
                        sections.push(DocumentSection {
                            id: Uuid::new_v4(),
                            document_id: Uuid::new_v4(),
                            content: current_section.trim().to_string(),
                            section_type,
                            order_index,
                            token_count: estimate_tokens(&current_section),
                            metadata: HashMap::new(),
                        });
                        order_index += 1;
                    }
                    current_section = line.to_string();
                    current_section.push('\n');
                    section_type = SectionType::Code;
                }
            } else {
                current_section.push_str(line);
                current_section.push('\n');
            }
        }
        
        // Save final section
        if !current_section.trim().is_empty() {
            sections.push(DocumentSection {
                id: Uuid::new_v4(),
                document_id: Uuid::new_v4(),
                content: current_section.trim().to_string(),
                section_type,
                order_index,
                token_count: estimate_tokens(&current_section),
                metadata: HashMap::new(),
            });
        }
        
        Ok(sections)
    }
    
    /// Parse JSON documents
    fn parse_json(&self, content: &str) -> IngestionResult<Vec<DocumentSection>> {
        let json_value: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| IngestionError::FormatParsing(format!("Invalid JSON: {}", e)))?;
        
        let mut sections = Vec::new();
        self.extract_json_sections(&json_value, "", &mut sections, 0);
        Ok(sections)
    }
    
    /// Parse CSV files
    fn parse_csv(&self, content: &str) -> IngestionResult<Vec<DocumentSection>> {
        let mut sections = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        
        if lines.is_empty() {
            return Ok(sections);
        }
        
        // Parse header
        let header = lines[0];
        sections.push(DocumentSection {
            id: Uuid::new_v4(),
            document_id: Uuid::new_v4(),
            content: header.to_string(),
            section_type: SectionType::Header,
            order_index: 0,
            token_count: estimate_tokens(header),
            metadata: HashMap::from([
                ("row_type".to_string(), "header".to_string()),
            ]),
        });
        
        // Parse data rows
        for (index, line) in lines.iter().skip(1).enumerate() {
            sections.push(DocumentSection {
                id: Uuid::new_v4(),
                document_id: Uuid::new_v4(),
                content: line.to_string(),
                section_type: SectionType::Table,
                order_index: index + 1,
                token_count: estimate_tokens(line),
                metadata: HashMap::from([
                    ("row_type".to_string(), "data".to_string()),
                    ("row_number".to_string(), (index + 1).to_string()),
                ]),
            });
        }
        
        Ok(sections)
    }
}

// ================================================================================================
// HELPER FUNCTIONS
// ================================================================================================

impl DocumentIngestionPipeline {
    fn extract_title(&self, sections: &[DocumentSection]) -> Option<String> {
        sections.iter()
            .filter(|s| s.section_type == SectionType::Header || s.section_type == SectionType::Title)
            .next()
            .map(|s| s.content.clone())
    }
    
    fn split_large_section(&self, content: &str, max_tokens: usize) -> Vec<String> {
        let sentences: Vec<&str> = content.split(". ").collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        
        for sentence in sentences {
            let test_chunk = if current_chunk.is_empty() {
                sentence.to_string()
            } else {
                format!("{}. {}", current_chunk, sentence)
            };
            
            if estimate_tokens(&test_chunk) > max_tokens && !current_chunk.is_empty() {
                chunks.push(current_chunk);
                current_chunk = sentence.to_string();
            } else {
                current_chunk = test_chunk;
            }
        }
        
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }
        
        chunks
    }
    
    fn extract_json_sections(&self, value: &serde_json::Value, path: &str, sections: &mut Vec<DocumentSection>, order: usize) {
        match value {
            serde_json::Value::Object(obj) => {
                for (key, val) in obj {
                    let new_path = if path.is_empty() { key.clone() } else { format!("{}.{}", path, key) };
                    self.extract_json_sections(val, &new_path, sections, sections.len());
                }
            },
            serde_json::Value::Array(arr) => {
                for (idx, val) in arr.iter().enumerate() {
                    let new_path = format!("{}[{}]", path, idx);
                    self.extract_json_sections(val, &new_path, sections, sections.len());
                }
            },
            _ => {
                let content = format!("{}: {}", path, value);
                sections.push(DocumentSection {
                    id: Uuid::new_v4(),
                    document_id: Uuid::new_v4(),
                    content: content.clone(),
                    section_type: SectionType::Metadata,
                    order_index: order,
                    token_count: estimate_tokens(&content),
                    metadata: HashMap::from([
                        ("json_path".to_string(), path.to_string()),
                        ("json_type".to_string(), match value {
                            serde_json::Value::String(_) => "string",
                            serde_json::Value::Number(_) => "number",
                            serde_json::Value::Bool(_) => "boolean",
                            serde_json::Value::Null => "null",
                            _ => "unknown"
                        }.to_string()),
                    ]),
                });
            }
        }
    }
}

/// Simple token estimation (approximation: 4 characters = 1 token)
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

/// Extract programming language from code fence
fn extract_language_from_code_fence(line: &str) -> String {
    line.trim_start_matches("```")
        .split_whitespace()
        .next()
        .unwrap_or("unknown")
        .to_string()
}

// ================================================================================================
// FACTORY FUNCTION
// ================================================================================================

/// Create a new document ingestion pipeline with default configuration
pub fn create_document_pipeline() -> DocumentIngestionPipeline {
    DocumentIngestionPipeline::with_default_config()
}

/// Create a new document ingestion pipeline with custom configuration
pub fn create_document_pipeline_with_config(config: IngestionConfig) -> DocumentIngestionPipeline {
    DocumentIngestionPipeline::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_document_format_detection() {
        assert_eq!(DocumentFormat::from_extension("test.txt"), DocumentFormat::PlainText);
        assert_eq!(DocumentFormat::from_extension("README.md"), DocumentFormat::Markdown);
        assert_eq!(DocumentFormat::from_extension("data.json"), DocumentFormat::Json);
        assert_eq!(DocumentFormat::from_extension("data.csv"), DocumentFormat::Csv);
        assert_eq!(DocumentFormat::from_extension("unknown.xyz"), DocumentFormat::Unknown);
    }
    
    #[test]
    fn test_token_estimation() {
        assert_eq!(estimate_tokens("hello"), 2);
        assert_eq!(estimate_tokens("hello world"), 3);
        assert_eq!(estimate_tokens(""), 1);
    }
    
    #[tokio::test]
    async fn test_plain_text_parsing() {
        let mut pipeline = DocumentIngestionPipeline::with_default_config();
        let content = "First paragraph.\n\nSecond paragraph with more content.";
        
        let raw_doc = RawDocument::new(content.to_string(), DocumentFormat::PlainText);
        let result = pipeline.process_document(raw_doc).await.unwrap();
        
        assert_eq!(result.sections.len(), 2);
        assert_eq!(result.sections[0].section_type, SectionType::Paragraph);
        assert_eq!(result.sections[1].section_type, SectionType::Paragraph);
    }
}