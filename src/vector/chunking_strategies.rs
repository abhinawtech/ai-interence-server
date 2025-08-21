// ================================================================================================
// DAY 10.2: INTELLIGENT CHUNKING STRATEGIES
// ================================================================================================
//
// Advanced text chunking algorithms that:
// - Preserve semantic boundaries
// - Handle overlapping chunks for better retrieval
// - Adapt chunk size based on content type
// - Maintain context across chunks
// - Optimize for search and retrieval performance
//
// ================================================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;
use tracing::{info, debug};

// ================================================================================================
// CHUNKING CONFIGURATION
// ================================================================================================

/// Different chunking strategies available
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChunkingStrategy {
    /// Fixed size chunks (simple but may break context)
    FixedSize { size: usize },
    
    /// Semantic boundary chunking (preserves meaning)
    Semantic { 
        target_size: usize,
        boundary_types: Vec<BoundaryType>,
    },
    
    /// Sliding window with overlap
    SlidingWindow { 
        size: usize, 
        overlap: usize,
    },
    
    /// Hierarchical chunking (nested structure)
    Hierarchical {
        levels: Vec<usize>, // e.g., [2000, 500] for 2-level hierarchy
        overlap_ratio: f32,
    },
    
    /// Content-adaptive chunking
    Adaptive {
        min_size: usize,
        max_size: usize,
        content_type_weights: HashMap<String, f32>,
    },
}

/// Types of semantic boundaries to respect
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BoundaryType {
    Sentence,
    Paragraph,
    Section,
    CodeBlock,
    ListItem,
    Table,
}

/// Configuration for chunking operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub strategy: ChunkingStrategy,
    pub preserve_metadata: bool,
    pub add_context_headers: bool,
    pub min_chunk_size: usize,
    pub max_chunk_size: usize,
    pub overlap_metadata: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::Semantic { 
                target_size: 500,
                boundary_types: vec![BoundaryType::Paragraph, BoundaryType::Sentence],
            },
            preserve_metadata: true,
            add_context_headers: true,
            min_chunk_size: 50,
            max_chunk_size: 2000,
            overlap_metadata: true,
        }
    }
}

// ================================================================================================
// CHUNK TYPES
// ================================================================================================

/// A text chunk with metadata and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    pub id: Uuid,
    pub document_id: Uuid,
    pub content: String,
    pub start_position: usize,
    pub end_position: usize,
    pub chunk_index: usize,
    pub token_count: usize,
    pub chunk_type: ChunkType,
    pub boundary_info: BoundaryInfo,
    pub overlap_info: Option<OverlapInfo>,
    pub context_header: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl TextChunk {
    pub fn new(
        document_id: Uuid,
        content: String,
        start_pos: usize,
        chunk_index: usize,
    ) -> Self {
        let token_count = estimate_tokens(&content);
        let end_pos = start_pos + content.len();
        
        Self {
            id: Uuid::new_v4(),
            document_id,
            content,
            start_position: start_pos,
            end_position: end_pos,
            chunk_index,
            token_count,
            chunk_type: ChunkType::Content,
            boundary_info: BoundaryInfo::default(),
            overlap_info: None,
            context_header: None,
            metadata: HashMap::new(),
        }
    }
}

/// Types of chunks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChunkType {
    Content,      // Regular content chunk
    Header,       // Section or chapter header
    Code,         // Code block
    Table,        // Table data
    List,         // List items
    Metadata,     // Document metadata
    Overlap,      // Overlapping context chunk
}

/// Information about chunk boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryInfo {
    pub starts_at_boundary: bool,
    pub ends_at_boundary: bool,
    pub boundary_types: Vec<BoundaryType>,
    pub confidence: f32,
}

impl Default for BoundaryInfo {
    fn default() -> Self {
        Self {
            starts_at_boundary: false,
            ends_at_boundary: false,
            boundary_types: Vec::new(),
            confidence: 0.0,
        }
    }
}

/// Information about chunk overlaps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapInfo {
    pub overlaps_with_previous: bool,
    pub overlaps_with_next: bool,
    pub overlap_tokens: usize,
    pub overlap_content: String,
}

/// Results of chunking operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingResult {
    pub document_id: Uuid,
    pub chunks: Vec<TextChunk>,
    pub total_chunks: usize,
    pub total_tokens: usize,
    pub average_chunk_size: f32,
    pub strategy_used: ChunkingStrategy,
    pub processing_time_ms: u64,
    pub quality_metrics: ChunkingQualityMetrics,
}

/// Quality metrics for chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingQualityMetrics {
    pub boundary_preservation_score: f32,
    pub size_consistency_score: f32,
    pub overlap_coverage_score: f32,
    pub context_preservation_score: f32,
}

// ================================================================================================
// ERROR HANDLING
// ================================================================================================

#[derive(Debug, Error)]
pub enum ChunkingError {
    #[error("Invalid chunk size: {0}")]
    InvalidChunkSize(usize),
    
    #[error("Content too small to chunk: {size} < {min}")]
    ContentTooSmall { size: usize, min: usize },
    
    #[error("Chunking strategy failed: {0}")]
    StrategyFailed(String),
    
    #[error("Boundary detection failed: {0}")]
    BoundaryDetectionFailed(String),
}

pub type ChunkingResult2<T> = Result<T, ChunkingError>;

// ================================================================================================
// INTELLIGENT CHUNKING ENGINE
// ================================================================================================

/// Main chunking engine with multiple strategies
pub struct IntelligentChunker {
    config: ChunkingConfig,
}

impl IntelligentChunker {
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }
    
    pub fn with_default_config() -> Self {
        Self::new(ChunkingConfig::default())
    }
    
    /// Chunk a document using the configured strategy
    pub fn chunk_document(&self, document_id: Uuid, content: &str) -> ChunkingResult2<ChunkingResult> {
        info!("🔪 Starting intelligent chunking for document: {}", document_id);
        let start_time = std::time::Instant::now();
        
        // Validate input
        if content.len() < self.config.min_chunk_size {
            return Err(ChunkingError::ContentTooSmall {
                size: content.len(),
                min: self.config.min_chunk_size,
            });
        }
        
        // Apply chunking strategy
        let chunks = match &self.config.strategy {
            ChunkingStrategy::FixedSize { size } => {
                self.fixed_size_chunking(document_id, content, *size)?
            },
            ChunkingStrategy::Semantic { target_size, boundary_types } => {
                self.semantic_chunking(document_id, content, *target_size, boundary_types)?
            },
            ChunkingStrategy::SlidingWindow { size, overlap } => {
                self.sliding_window_chunking(document_id, content, *size, *overlap)?
            },
            ChunkingStrategy::Hierarchical { levels, overlap_ratio } => {
                self.hierarchical_chunking(document_id, content, levels, *overlap_ratio)?
            },
            ChunkingStrategy::Adaptive { min_size, max_size, content_type_weights } => {
                self.adaptive_chunking(document_id, content, *min_size, *max_size, content_type_weights)?
            },
        };
        
        // Calculate metrics
        let total_tokens: usize = chunks.iter().map(|c| c.token_count).sum();
        let average_chunk_size = chunks.len() as f32 / chunks.len() as f32;
        let quality_metrics = self.calculate_quality_metrics(&chunks, content);
        
        let result = ChunkingResult {
            document_id,
            total_chunks: chunks.len(),
            total_tokens,
            average_chunk_size,
            chunks,
            strategy_used: self.config.strategy.clone(),
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            quality_metrics,
        };
        
        info!("✅ Chunking complete: {} chunks, {} tokens, {}ms", 
              result.total_chunks, result.total_tokens, result.processing_time_ms);
        
        Ok(result)
    }
}

// ================================================================================================
// CHUNKING STRATEGY IMPLEMENTATIONS
// ================================================================================================

impl IntelligentChunker {
    /// Simple fixed-size chunking
    fn fixed_size_chunking(&self, document_id: Uuid, content: &str, chunk_size: usize) -> ChunkingResult2<Vec<TextChunk>> {
        debug!("Applying fixed-size chunking with size: {}", chunk_size);
        
        let mut chunks = Vec::new();
        let mut start_pos = 0;
        let mut chunk_index = 0;
        
        while start_pos < content.len() {
            let end_pos = std::cmp::min(start_pos + chunk_size, content.len());
            let chunk_content = content[start_pos..end_pos].to_string();
            
            let mut chunk = TextChunk::new(document_id, chunk_content, start_pos, chunk_index);
            chunk.chunk_type = ChunkType::Content;
            
            if self.config.preserve_metadata {
                chunk.metadata.insert("strategy".to_string(), "fixed_size".to_string());
                chunk.metadata.insert("target_size".to_string(), chunk_size.to_string());
            }
            
            chunks.push(chunk);
            start_pos = end_pos;
            chunk_index += 1;
        }
        
        Ok(chunks)
    }
    
    /// Semantic boundary-aware chunking
    fn semantic_chunking(&self, document_id: Uuid, content: &str, target_size: usize, boundary_types: &[BoundaryType]) -> ChunkingResult2<Vec<TextChunk>> {
        debug!("Applying semantic chunking with target size: {} and boundaries: {:?}", target_size, boundary_types);
        
        let boundaries = self.detect_boundaries(content, boundary_types);
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut start_pos = 0;
        let mut chunk_index = 0;
        
        for boundary in boundaries {
            let segment = &content[start_pos..boundary.position];
            
            // Check if adding this segment would exceed target size
            if !current_chunk.is_empty() && estimate_tokens(&(current_chunk.clone() + segment)) > target_size {
                // Create chunk from accumulated content
                if !current_chunk.is_empty() {
                    let mut chunk = TextChunk::new(document_id, current_chunk.clone(), start_pos - current_chunk.len(), chunk_index);
                    chunk.boundary_info = BoundaryInfo {
                        starts_at_boundary: true,
                        ends_at_boundary: true,
                        boundary_types: boundary_types.to_vec(),
                        confidence: boundary.confidence,
                    };
                    
                    if self.config.preserve_metadata {
                        chunk.metadata.insert("strategy".to_string(), "semantic".to_string());
                        chunk.metadata.insert("boundary_score".to_string(), boundary.confidence.to_string());
                    }
                    
                    chunks.push(chunk);
                    chunk_index += 1;
                    current_chunk.clear();
                }
            }
            
            current_chunk.push_str(segment);
            start_pos = boundary.position;
        }
        
        // Handle remaining content
        if !current_chunk.is_empty() {
            let mut chunk = TextChunk::new(document_id, current_chunk.clone(), start_pos - current_chunk.len(), chunk_index);
            chunk.boundary_info.ends_at_boundary = true;
            chunks.push(chunk);
        }
        
        Ok(chunks)
    }
    
    /// Sliding window chunking with overlap
    fn sliding_window_chunking(&self, document_id: Uuid, content: &str, size: usize, overlap: usize) -> ChunkingResult2<Vec<TextChunk>> {
        debug!("Applying sliding window chunking with size: {}, overlap: {}", size, overlap);
        
        if overlap >= size {
            return Err(ChunkingError::InvalidChunkSize(overlap));
        }
        
        let mut chunks = Vec::new();
        let mut start_pos = 0;
        let mut chunk_index = 0;
        let step_size = size - overlap;
        
        while start_pos < content.len() {
            let end_pos = std::cmp::min(start_pos + size, content.len());
            let chunk_content = content[start_pos..end_pos].to_string();
            
            let mut chunk = TextChunk::new(document_id, chunk_content, start_pos, chunk_index);
            
            // Add overlap information
            if chunk_index > 0 {
                let overlap_start = std::cmp::max(0, start_pos as i32 - overlap as i32) as usize;
                let overlap_content = content[overlap_start..start_pos].to_string();
                
                chunk.overlap_info = Some(OverlapInfo {
                    overlaps_with_previous: true,
                    overlaps_with_next: start_pos + size < content.len(),
                    overlap_tokens: estimate_tokens(&overlap_content),
                    overlap_content,
                });
            }
            
            if self.config.preserve_metadata {
                chunk.metadata.insert("strategy".to_string(), "sliding_window".to_string());
                chunk.metadata.insert("overlap_size".to_string(), overlap.to_string());
            }
            
            chunks.push(chunk);
            
            // Break if we've reached the end
            if end_pos >= content.len() {
                break;
            }
            
            start_pos += step_size;
            chunk_index += 1;
        }
        
        Ok(chunks)
    }
    
    /// Hierarchical chunking with multiple levels
    fn hierarchical_chunking(&self, document_id: Uuid, content: &str, levels: &[usize], overlap_ratio: f32) -> ChunkingResult2<Vec<TextChunk>> {
        debug!("Applying hierarchical chunking with levels: {:?}", levels);
        
        if levels.is_empty() {
            return Err(ChunkingError::StrategyFailed("No levels specified".to_string()));
        }
        
        let mut all_chunks = Vec::new();
        
        // Process each hierarchical level
        for (level_idx, &level_size) in levels.iter().enumerate() {
            let overlap = (level_size as f32 * overlap_ratio) as usize;
            
            // Create sliding window chunks for this level
            let level_chunks = self.sliding_window_chunking(document_id, content, level_size, overlap)?;
            
            // Add level information to chunks
            for mut chunk in level_chunks {
                chunk.metadata.insert("hierarchy_level".to_string(), level_idx.to_string());
                chunk.metadata.insert("level_size".to_string(), level_size.to_string());
                chunk.metadata.insert("strategy".to_string(), "hierarchical".to_string());
                all_chunks.push(chunk);
            }
        }
        
        Ok(all_chunks)
    }
    
    /// Adaptive chunking based on content analysis
    fn adaptive_chunking(&self, document_id: Uuid, content: &str, min_size: usize, max_size: usize, _weights: &HashMap<String, f32>) -> ChunkingResult2<Vec<TextChunk>> {
        debug!("Applying adaptive chunking with size range: {}-{}", min_size, max_size);
        
        // Analyze content characteristics
        let content_analysis = self.analyze_content(content);
        
        // Determine optimal chunk size based on content
        let optimal_size = self.calculate_optimal_chunk_size(&content_analysis, min_size, max_size);
        
        // Apply semantic chunking with the optimal size
        self.semantic_chunking(
            document_id, 
            content, 
            optimal_size, 
            &[BoundaryType::Paragraph, BoundaryType::Sentence]
        )
    }
}

// ================================================================================================
// BOUNDARY DETECTION
// ================================================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SemanticBoundary {
    position: usize,
    #[allow(dead_code)]
    boundary_type: BoundaryType,
    confidence: f32,
}

impl IntelligentChunker {
    fn detect_boundaries(&self, content: &str, boundary_types: &[BoundaryType]) -> Vec<SemanticBoundary> {
        let mut boundaries = Vec::new();
        
        for &ref boundary_type in boundary_types {
            match boundary_type {
                BoundaryType::Paragraph => {
                    self.detect_paragraph_boundaries(content, &mut boundaries);
                },
                BoundaryType::Sentence => {
                    self.detect_sentence_boundaries(content, &mut boundaries);
                },
                BoundaryType::Section => {
                    self.detect_section_boundaries(content, &mut boundaries);
                },
                BoundaryType::CodeBlock => {
                    self.detect_code_boundaries(content, &mut boundaries);
                },
                _ => {} // TODO: Implement other boundary types
            }
        }
        
        // Sort by position
        boundaries.sort_by_key(|b| b.position);
        boundaries
    }
    
    fn detect_paragraph_boundaries(&self, content: &str, boundaries: &mut Vec<SemanticBoundary>) {
        for (pos, _) in content.match_indices("\n\n") {
            boundaries.push(SemanticBoundary {
                position: pos,
                boundary_type: BoundaryType::Paragraph,
                confidence: 0.9,
            });
        }
    }
    
    fn detect_sentence_boundaries(&self, content: &str, boundaries: &mut Vec<SemanticBoundary>) {
        for (pos, _) in content.match_indices(". ") {
            boundaries.push(SemanticBoundary {
                position: pos + 2,
                boundary_type: BoundaryType::Sentence,
                confidence: 0.7,
            });
        }
    }
    
    fn detect_section_boundaries(&self, content: &str, boundaries: &mut Vec<SemanticBoundary>) {
        // Detect markdown headers
        for line in content.lines() {
            if line.starts_with('#') {
                if let Some(pos) = content.find(line) {
                    boundaries.push(SemanticBoundary {
                        position: pos,
                        boundary_type: BoundaryType::Section,
                        confidence: 0.95,
                    });
                }
            }
        }
    }
    
    fn detect_code_boundaries(&self, content: &str, boundaries: &mut Vec<SemanticBoundary>) {
        for (pos, _) in content.match_indices("```") {
            boundaries.push(SemanticBoundary {
                position: pos,
                boundary_type: BoundaryType::CodeBlock,
                confidence: 0.85,
            });
        }
    }
}

// ================================================================================================
// CONTENT ANALYSIS AND OPTIMIZATION
// ================================================================================================

#[derive(Debug)]
#[allow(dead_code)]
struct ContentAnalysis {
    avg_sentence_length: f32,
    #[allow(dead_code)]
    paragraph_count: usize,
    #[allow(dead_code)]
    code_block_ratio: f32,
    #[allow(dead_code)]
    list_item_count: usize,
    complexity_score: f32,
}

impl IntelligentChunker {
    fn analyze_content(&self, content: &str) -> ContentAnalysis {
        let sentences: Vec<&str> = content.split(". ").collect();
        let paragraphs: Vec<&str> = content.split("\n\n").collect();
        let code_blocks = content.matches("```").count() / 2;
        let list_items = content.matches("\n- ").count() + content.matches("\n* ").count();
        
        let avg_sentence_length = if !sentences.is_empty() {
            sentences.iter().map(|s| s.len()).sum::<usize>() as f32 / sentences.len() as f32
        } else {
            0.0
        };
        
        let code_block_ratio = if content.len() > 0 {
            code_blocks as f32 / content.len() as f32
        } else {
            0.0
        };
        
        // Simple complexity score based on various factors
        let complexity_score = (avg_sentence_length / 50.0) + 
                              (code_block_ratio * 2.0) + 
                              (list_items as f32 / paragraphs.len() as f32);
        
        ContentAnalysis {
            avg_sentence_length,
            paragraph_count: paragraphs.len(),
            code_block_ratio,
            list_item_count: list_items,
            complexity_score,
        }
    }
    
    fn calculate_optimal_chunk_size(&self, analysis: &ContentAnalysis, min_size: usize, max_size: usize) -> usize {
        let base_size = (min_size + max_size) / 2;
        
        // Adjust based on content characteristics
        let size_modifier = if analysis.complexity_score > 1.0 {
            0.8 // Smaller chunks for complex content
        } else if analysis.avg_sentence_length > 100.0 {
            1.2 // Larger chunks for verbose content
        } else {
            1.0
        };
        
        let optimal_size = (base_size as f32 * size_modifier) as usize;
        
        // Ensure within bounds
        optimal_size.max(min_size).min(max_size)
    }
    
    fn calculate_quality_metrics(&self, chunks: &[TextChunk], _original_content: &str) -> ChunkingQualityMetrics {
        let boundary_score = chunks.iter()
            .map(|c| c.boundary_info.confidence)
            .sum::<f32>() / chunks.len() as f32;
        
        let sizes: Vec<usize> = chunks.iter().map(|c| c.token_count).collect();
        let avg_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        let size_variance = sizes.iter()
            .map(|&s| (s as f32 - avg_size).powi(2))
            .sum::<f32>() / sizes.len() as f32;
        let size_consistency = 1.0 - (size_variance.sqrt() / avg_size).min(1.0);
        
        let overlap_coverage = chunks.iter()
            .filter(|c| c.overlap_info.is_some())
            .count() as f32 / chunks.len() as f32;
        
        ChunkingQualityMetrics {
            boundary_preservation_score: boundary_score,
            size_consistency_score: size_consistency,
            overlap_coverage_score: overlap_coverage,
            context_preservation_score: 0.8, // Placeholder
        }
    }
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

/// Simple token estimation (4 chars ≈ 1 token)
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

// ================================================================================================
// FACTORY FUNCTIONS
// ================================================================================================

/// Create a chunker with semantic strategy
pub fn create_semantic_chunker(target_size: usize) -> IntelligentChunker {
    let config = ChunkingConfig {
        strategy: ChunkingStrategy::Semantic {
            target_size,
            boundary_types: vec![BoundaryType::Paragraph, BoundaryType::Sentence],
        },
        ..Default::default()
    };
    IntelligentChunker::new(config)
}

/// Create a chunker with sliding window strategy
pub fn create_sliding_window_chunker(size: usize, overlap: usize) -> IntelligentChunker {
    let config = ChunkingConfig {
        strategy: ChunkingStrategy::SlidingWindow { size, overlap },
        ..Default::default()
    };
    IntelligentChunker::new(config)
}

/// Create an adaptive chunker
pub fn create_adaptive_chunker(min_size: usize, max_size: usize) -> IntelligentChunker {
    let config = ChunkingConfig {
        strategy: ChunkingStrategy::Adaptive {
            min_size,
            max_size,
            content_type_weights: HashMap::new(),
        },
        ..Default::default()
    };
    IntelligentChunker::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fixed_size_chunking() {
        let chunker = IntelligentChunker::new(ChunkingConfig {
            strategy: ChunkingStrategy::FixedSize { size: 100 },
            ..Default::default()
        });
        
        let content = "This is a test document. ".repeat(10);
        let doc_id = Uuid::new_v4();
        let result = chunker.chunk_document(doc_id, &content).unwrap();
        
        assert!(result.total_chunks > 1);
        assert!(result.chunks.iter().all(|c| c.content.len() <= 100));
    }
    
    #[test]
    fn test_semantic_chunking() {
        let chunker = create_semantic_chunker(200);
        
        let content = "First paragraph.\n\nSecond paragraph with more content.\n\nThird paragraph.";
        let doc_id = Uuid::new_v4();
        let result = chunker.chunk_document(doc_id, content).unwrap();
        
        assert!(result.total_chunks >= 1);
        assert!(result.chunks.iter().all(|c| c.boundary_info.confidence > 0.0));
    }
}