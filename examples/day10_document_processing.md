# Day 10: Document Processing Pipeline Examples

This guide demonstrates Day 10's comprehensive document processing capabilities including intelligent ingestion, chunking, and incremental updates with deduplication.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw Documents  │ -> │  Ingestion       │ -> │  Processed      │
│  • Text         │    │  Pipeline        │    │  Documents      │
│  • Markdown     │    │  • Format Parse  │    │  • Sections     │
│  • JSON         │    │  • Metadata      │    │  • Metadata     │
│  • CSV          │    │  • Structure     │    │  • Tokens       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Incremental    │ <- │  Intelligent     │ <- │  Text Chunks    │
│  Updates        │    │  Chunking        │    │  • Semantic     │
│  • Versions     │    │  • Boundaries    │    │  • Overlapping  │
│  • Changes      │    │  • Overlaps      │    │  • Hierarchical │
│  • Dedup        │    │  • Adaptive      │    │  • Context      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Day 10.1: Document Ingestion and Parsing Pipeline

### Features

1. **Multi-Format Support**: Text, Markdown, JSON, CSV with extensible format detection
2. **Intelligent Parsing**: Structure-aware parsing that preserves semantic meaning
3. **Metadata Extraction**: Automatic extraction of document properties and structure
4. **Batch Processing**: Efficient processing of multiple documents
5. **Progress Tracking**: Real-time processing statistics and error handling

### Example: Ingesting Different Document Formats

```rust
use ai_inference_server::vector::{create_document_pipeline, RawDocument, DocumentFormat};

// Create ingestion pipeline
let mut pipeline = create_document_pipeline();

// Ingest Markdown document
let markdown_content = r#"
# AI Research Paper

## Introduction
This paper explores the latest developments in artificial intelligence.

## Methodology
We used the following approaches:
- Deep learning models
- Transformer architectures
- Large-scale training datasets

```python
def train_model(data):
    model = Transformer(layers=12)
    return model.fit(data)
```

## Results
Our model achieved 95% accuracy on the benchmark dataset.
"#;

let processed = pipeline.process_document(
    RawDocument::new(markdown_content.to_string(), DocumentFormat::Markdown)
).await?;

println!("Processed {} sections with {} total tokens", 
         processed.sections.len(), processed.total_tokens);
```

### Example: Batch Processing Files

```rust
// Process multiple files in batch
let file_paths = vec![
    "docs/paper1.md".to_string(),
    "data/dataset.csv".to_string(),
    "config/settings.json".to_string(),
];

let results = pipeline.ingest_batch(file_paths).await;

for (i, result) in results.iter().enumerate() {
    match result {
        Ok(doc) => println!("✅ Document {}: {} sections", i, doc.sections.len()),
        Err(e) => println!("❌ Document {}: Error - {}", i, e),
    }
}

// Get processing statistics
let stats = pipeline.get_stats();
println!("Processed {} documents in {}ms", 
         stats.documents_processed, stats.processing_time_ms);
```

## 🧠 Day 10.2: Intelligent Chunking Strategies

### Features

1. **Semantic Boundaries**: Respects paragraph, sentence, and section boundaries
2. **Sliding Window**: Creates overlapping chunks for better context retrieval
3. **Hierarchical Chunking**: Multi-level chunks for different granularities
4. **Adaptive Sizing**: Adjusts chunk size based on content complexity
5. **Quality Metrics**: Evaluates chunking effectiveness

### Example: Semantic Chunking

```rust
use ai_inference_server::vector::{create_semantic_chunker, ChunkingStrategy};

let chunker = create_semantic_chunker(500); // 500 token target size

let document_content = r#"
Artificial Intelligence has revolutionized many industries. In healthcare, AI assists with diagnosis and treatment planning.

Machine learning algorithms can analyze vast amounts of medical data to identify patterns that humans might miss. This has led to earlier detection of diseases and more personalized treatment approaches.

The future of AI in medicine looks promising, with ongoing research into areas like drug discovery and robotic surgery.
"#;

let doc_id = uuid::Uuid::new_v4();
let result = chunker.chunk_document(doc_id, document_content)?;

for (i, chunk) in result.chunks.iter().enumerate() {
    println!("Chunk {}: {} tokens", i, chunk.token_count);
    println!("  Boundary confidence: {:.2}", chunk.boundary_info.confidence);
    println!("  Content preview: {}...", 
             chunk.content.chars().take(50).collect::<String>());
}

println!("Quality metrics:");
println!("  Boundary preservation: {:.2}", result.quality_metrics.boundary_preservation_score);
println!("  Size consistency: {:.2}", result.quality_metrics.size_consistency_score);
```

### Example: Sliding Window Chunking

```rust
use ai_inference_server::vector::create_sliding_window_chunker;

let chunker = create_sliding_window_chunker(
    300,  // chunk size
    100,  // overlap size
);

let result = chunker.chunk_document(doc_id, document_content)?;

for chunk in &result.chunks {
    if let Some(overlap_info) = &chunk.overlap_info {
        println!("Chunk {}: {} tokens, {} overlap tokens", 
                 chunk.chunk_index, 
                 chunk.token_count, 
                 overlap_info.overlap_tokens);
    }
}
```

### Example: Hierarchical Chunking

```rust
use ai_inference_server::vector::{ChunkingConfig, ChunkingStrategy};

let config = ChunkingConfig {
    strategy: ChunkingStrategy::Hierarchical {
        levels: vec![2000, 500],  // Large and small chunks
        overlap_ratio: 0.1,
    },
    preserve_metadata: true,
    add_context_headers: true,
    min_chunk_size: 50,
    max_chunk_size: 3000,
    overlap_metadata: true,
};

let chunker = IntelligentChunker::new(config);
let result = chunker.chunk_document(doc_id, document_content)?;

// Filter by hierarchy level
let large_chunks: Vec<_> = result.chunks.iter()
    .filter(|c| c.metadata.get("hierarchy_level") == Some(&"0".to_string()))
    .collect();

let small_chunks: Vec<_> = result.chunks.iter()
    .filter(|c| c.metadata.get("hierarchy_level") == Some(&"1".to_string()))
    .collect();

println!("Created {} large chunks and {} small chunks", 
         large_chunks.len(), small_chunks.len());
```

## 🔄 Day 10.3: Incremental Updates and Deduplication

### Features

1. **Change Detection**: Identifies what has changed between document versions
2. **Version Management**: Maintains complete version history with metadata
3. **Smart Deduplication**: Finds and merges similar content across documents
4. **Conflict Resolution**: Handles concurrent updates intelligently
5. **Storage Optimization**: Reduces storage through intelligent deduplication

### Example: Document Version Management

```rust
use ai_inference_server::vector::{create_incremental_manager, UpdateResult};

let mut manager = create_incremental_manager();

let doc_id = uuid::Uuid::new_v4();
let chunk_ids = vec![uuid::Uuid::new_v4()];

// Initial document creation
let content_v1 = "This is the original document content.";
let result = manager.update_document(doc_id, content_v1, chunk_ids.clone()).await?;

println!("Created version: {}", result.new_version);
println!("Change type: {:?}", result.change_type);

// Update the document
let content_v2 = "This is the updated document content with new information.";
let result = manager.update_document(doc_id, content_v2, chunk_ids).await?;

println!("Updated version: {} -> {}", 
         result.old_version.unwrap(), result.new_version);
println!("Similarity to previous: {:.2}", 
         result.change_summary.similarity_to_previous);

// View version history
if let Some(versions) = manager.get_version_history(doc_id) {
    for version in versions {
        println!("Version {}: {} sections, created at {}", 
                 version.version_number, 
                 version.change_summary.sections_added, 
                 version.created_at);
    }
}
```

### Example: Global Deduplication

```rust
// Find duplicate content across all documents
let candidates = manager.find_duplicates_global().await?;

for candidate in &candidates {
    println!("Found {} duplicates of chunk {}", 
             candidate.duplicate_chunk_ids.len(),
             candidate.primary_chunk_id);
    println!("  Similarity: {:.2}", candidate.similarity_score);
    println!("  Potential savings: {} bytes", 
             candidate.potential_savings.bytes_saved);
}

// Apply deduplication
let processed_chunks = manager.apply_deduplication(candidates).await?;
println!("Processed {} chunks for deduplication", processed_chunks.len());

// Get statistics
let stats = manager.get_stats();
println!("Statistics:");
println!("  Total documents: {}", stats.total_documents);
println!("  Total versions: {}", stats.total_versions);
println!("  Unique content hashes: {}", stats.unique_content_hashes);
println!("  Duplicate groups: {}", stats.duplicate_content_groups);
```

### Example: Change Detection and Analysis

```rust
use ai_inference_server::vector::{ContentFingerprint, ChangeType};

// Compare document versions
let old_content = "The quick brown fox jumps over the lazy dog.";
let new_content = "The quick brown fox jumps over the sleepy dog.";

let old_fingerprint = ContentFingerprint::from_content(old_content);
let new_fingerprint = ContentFingerprint::from_content(new_content);

// Fingerprints capture both content and structure
println!("Old hash: {}", old_fingerprint.content_hash);
println!("New hash: {}", new_fingerprint.content_hash);
println!("Structure changed: {}", 
         old_fingerprint.structure_hash != new_fingerprint.structure_hash);

// The update manager would analyze these changes
let change_summary = analyze_changes(&old_fingerprint, &new_fingerprint, new_content).await?;

match change_summary.change_type {
    ChangeType::MinorEdit => println!("Small change detected"),
    ChangeType::MajorEdit => println!("Significant change detected"),
    ChangeType::Rewrite => println!("Complete rewrite detected"),
    ChangeType::StructuralChange => println!("Document structure changed"),
    _ => {}
}
```

## 🎯 Performance and Quality Metrics

### Ingestion Pipeline Metrics

```rust
let stats = pipeline.get_stats();
println!("Ingestion Performance:");
println!("  Documents processed: {}", stats.documents_processed);
println!("  Total sections: {}", stats.total_sections);
println!("  Processing time: {}ms", stats.processing_time_ms);
println!("  Format breakdown:");
for (format, count) in &stats.format_breakdown {
    println!("    {:?}: {} documents", format, count);
}
```

### Chunking Quality Assessment

```rust
let metrics = &result.quality_metrics;
println!("Chunking Quality:");
println!("  Boundary preservation: {:.2}/1.0", metrics.boundary_preservation_score);
println!("  Size consistency: {:.2}/1.0", metrics.size_consistency_score);
println!("  Overlap coverage: {:.2}/1.0", metrics.overlap_coverage_score);
println!("  Context preservation: {:.2}/1.0", metrics.context_preservation_score);
```

### Storage Efficiency Analysis

```rust
let impact = &result.storage_impact;
println!("Storage Impact:");
println!("  Size change: {} bytes", impact.size_change_bytes);
println!("  Vector count change: {}", impact.vector_count_change);
println!("  Efficiency improvement: {:.1}%", impact.efficiency_improvement * 100.0);
```

## 🔧 Configuration and Customization

### Custom Ingestion Configuration

```rust
use ai_inference_server::vector::{IngestionConfig, create_document_pipeline_with_config};

let config = IngestionConfig {
    batch_size: 20,
    max_section_tokens: 1000,
    extract_metadata: true,
    preserve_structure: true,
    parallel_processing: false,  // Sequential for debugging
};

let pipeline = create_document_pipeline_with_config(config);
```

### Advanced Chunking Configuration

```rust
use ai_inference_server::vector::{ChunkingConfig, BoundaryType};

let config = ChunkingConfig {
    strategy: ChunkingStrategy::Semantic {
        target_size: 750,
        boundary_types: vec![
            BoundaryType::Section,
            BoundaryType::Paragraph,
            BoundaryType::Sentence,
            BoundaryType::CodeBlock,
        ],
    },
    preserve_metadata: true,
    add_context_headers: true,
    min_chunk_size: 100,
    max_chunk_size: 1500,
    overlap_metadata: true,
};
```

### Incremental Update Configuration

```rust
use ai_inference_server::vector::{IncrementalUpdateConfig, ConflictResolutionStrategy};

let config = IncrementalUpdateConfig {
    similarity_threshold: 0.90,
    deduplication_threshold: 0.95,
    max_versions_to_keep: 20,
    enable_structural_diff: true,
    batch_update_size: 50,
    conflict_resolution: ConflictResolutionStrategy::MergeChanges,
};

let manager = create_incremental_manager_with_config(config);
```

## 🏆 Best Practices

### 1. Format-Specific Optimization
- Use appropriate parsers for each document format
- Preserve structural information (headers, lists, code blocks)
- Extract format-specific metadata

### 2. Chunking Strategy Selection
- **Semantic**: Best for documents with clear structure
- **Sliding Window**: Ideal for continuous text and retrieval systems
- **Hierarchical**: Perfect for documents with multiple levels of detail
- **Adaptive**: Use when content varies significantly in complexity

### 3. Version Management
- Regular cleanup of old versions based on retention policy
- Monitor storage growth and deduplication effectiveness
- Use structural diffing for better change detection

### 4. Performance Optimization
- Batch processing for multiple documents
- Parallel processing when system resources allow
- Cache fingerprints for frequently accessed documents
- Regular deduplication runs to optimize storage

## 📊 Integration with Vector Search

Day 10's processed documents integrate seamlessly with the vector database:

```rust
// After processing and chunking, store in vector database
for chunk in result.chunks {
    let embedding = embedding_service.generate_embedding(&chunk.content).await?;
    
    let vector_point = VectorPoint::with_metadata(
        embedding.vector,
        chunk.metadata.clone(),
    );
    
    vector_storage.store_vector(vector_point).await?;
}
```

This creates a comprehensive document processing pipeline that transforms raw documents into searchable, intelligently chunked vectors with full version control and deduplication capabilities.