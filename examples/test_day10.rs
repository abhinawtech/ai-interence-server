// ================================================================================================
// DAY 10 FUNCTIONALITY TEST
// ================================================================================================
//
// Comprehensive test of Day 10 document processing features:
// - Document ingestion and parsing
// - Intelligent chunking strategies  
// - Incremental updates and deduplication
//
// ================================================================================================

use uuid::Uuid;
use std::collections::HashMap;

use ai_interence_server::vector::{
    // Document ingestion
    create_document_pipeline, RawDocument, DocumentFormat, IngestionConfig,
    // Chunking
    create_semantic_chunker, create_sliding_window_chunker, create_adaptive_chunker,
    ChunkingStrategy, ChunkingConfig, BoundaryType,
    // Incremental updates
    create_incremental_manager, IncrementalUpdateConfig, ContentFingerprint,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Day 10: Document Processing Pipeline Tests");
    println!("=" .repeat(60));
    
    // Test 1: Document Ingestion
    test_document_ingestion().await?;
    println!();
    
    // Test 2: Intelligent Chunking
    test_intelligent_chunking().await?;
    println!();
    
    // Test 3: Incremental Updates
    test_incremental_updates().await?;
    println!();
    
    // Test 4: Deduplication
    test_deduplication().await?;
    println!();
    
    println!("✅ All Day 10 tests completed successfully!");
    
    Ok(())
}

// ================================================================================================
// TEST 1: DOCUMENT INGESTION AND PARSING
// ================================================================================================

async fn test_document_ingestion() -> Result<(), Box<dyn std::error::Error>> {
    println!("📄 Test 1: Document Ingestion and Parsing Pipeline");
    println!("-".repeat(50));
    
    let mut pipeline = create_document_pipeline();
    
    // Test different document formats
    let test_documents = vec![
        ("Plain Text", DocumentFormat::PlainText, r#"
This is a plain text document.

It has multiple paragraphs with different content.

The pipeline should automatically detect paragraph boundaries and create sections accordingly.
        "#.trim()),
        
        ("Markdown", DocumentFormat::Markdown, r#"
# AI Research Paper

## Introduction
This paper explores artificial intelligence developments.

### Key Contributions
- Novel transformer architecture
- Improved training efficiency 
- Better performance metrics

## Methodology
We used the following approach:

```python
def train_model(data):
    model = Transformer(layers=12)
    return model.fit(data)
```

## Results
Our model achieved **95% accuracy** on benchmarks.
        "#.trim()),
        
        ("JSON", DocumentFormat::Json, r#"
{
    "title": "Configuration File",
    "version": "1.0",
    "settings": {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "features": [
        "semantic_search",
        "document_processing",
        "vector_storage"
    ]
}
        "#.trim()),
        
        ("CSV", DocumentFormat::Csv, r#"
name,age,department,role
Alice Johnson,28,Engineering,Senior Developer
Bob Smith,35,Product,Product Manager
Carol Davis,31,Engineering,Tech Lead
David Wilson,29,Design,UX Designer
        "#.trim()),
    ];
    
    for (name, format, content) in test_documents {
        println!("  Processing {} document...", name);
        
        let raw_doc = RawDocument::new(content.to_string(), format);
        let processed = pipeline.process_document(raw_doc).await?;
        
        println!("    ✅ Processed: {} sections, {} tokens", 
                 processed.sections.len(), processed.total_tokens);
        
        // Show first section as example
        if !processed.sections.is_empty() {
            let first_section = &processed.sections[0];
            let preview = first_section.content.chars().take(50).collect::<String>();
            println!("    📝 First section: {}...", preview);
        }
    }
    
    let stats = pipeline.get_stats();
    println!("  📊 Final Statistics:");
    println!("    Documents processed: {}", stats.documents_processed);
    println!("    Total sections: {}", stats.total_sections);
    println!("    Total tokens: {}", stats.total_tokens);
    
    Ok(())
}

// ================================================================================================
// TEST 2: INTELLIGENT CHUNKING STRATEGIES
// ================================================================================================

async fn test_intelligent_chunking() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Test 2: Intelligent Chunking Strategies");
    println!("-".repeat(50));
    
    let test_content = r#"
Artificial Intelligence has revolutionized many industries. In healthcare, AI assists with diagnosis and treatment planning. Machine learning algorithms can analyze vast amounts of medical data to identify patterns that humans might miss.

This breakthrough has led to earlier detection of diseases and more personalized treatment approaches. Doctors can now leverage AI to make more informed decisions about patient care.

The future of AI in medicine looks promising. Ongoing research focuses on drug discovery, robotic surgery, and predictive analytics. These advances could transform how we approach healthcare in the coming decades.

Natural language processing allows AI systems to understand medical literature and patient records. Computer vision helps with medical imaging analysis. Together, these technologies create powerful diagnostic tools.
    "#.trim();
    
    let doc_id = Uuid::new_v4();
    
    // Test 1: Semantic Chunking
    println!("  🎯 Testing Semantic Chunking (500 tokens)...");
    let semantic_chunker = create_semantic_chunker(500);
    let result = semantic_chunker.chunk_document(doc_id, test_content)?;
    
    println!("    Created {} chunks", result.total_chunks);
    println!("    Quality - Boundary: {:.2}, Consistency: {:.2}", 
             result.quality_metrics.boundary_preservation_score,
             result.quality_metrics.size_consistency_score);
    
    for (i, chunk) in result.chunks.iter().take(2).enumerate() {
        println!("    Chunk {}: {} tokens, confidence: {:.2}", 
                 i, chunk.token_count, chunk.boundary_info.confidence);
    }
    
    // Test 2: Sliding Window Chunking  
    println!("  🔄 Testing Sliding Window Chunking (200 size, 50 overlap)...");
    let window_chunker = create_sliding_window_chunker(200, 50);
    let result = window_chunker.chunk_document(doc_id, test_content)?;
    
    println!("    Created {} chunks with overlaps", result.total_chunks);
    let overlap_count = result.chunks.iter().filter(|c| c.overlap_info.is_some()).count();
    println!("    Chunks with overlap: {}", overlap_count);
    
    // Test 3: Adaptive Chunking
    println!("  🎛️  Testing Adaptive Chunking (100-800 range)...");
    let adaptive_chunker = create_adaptive_chunker(100, 800);
    let result = adaptive_chunker.chunk_document(doc_id, test_content)?;
    
    println!("    Created {} adaptive chunks", result.total_chunks);
    println!("    Average size: {:.1} tokens", result.average_chunk_size);
    
    Ok(())
}

// ================================================================================================
// TEST 3: INCREMENTAL UPDATES AND VERSION MANAGEMENT
// ================================================================================================

async fn test_incremental_updates() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Test 3: Incremental Updates and Version Management");
    println!("-".repeat(50));
    
    let mut manager = create_incremental_manager();
    let doc_id = Uuid::new_v4();
    let chunk_ids = vec![Uuid::new_v4(), Uuid::new_v4()];
    
    // Version 1: Initial document
    println!("  📝 Creating initial document version...");
    let content_v1 = "This is the original document about machine learning. It covers basic concepts and applications.";
    
    let result = manager.update_document(doc_id, content_v1, chunk_ids.clone()).await?;
    println!("    ✅ Created version: {}", result.new_version);
    println!("    Change type: {:?}", result.change_summary.change_type);
    
    // Version 2: Minor update
    println!("  ✏️  Making minor updates...");
    let content_v2 = "This is the updated document about machine learning. It covers basic concepts, applications, and recent advances.";
    
    let result = manager.update_document(doc_id, content_v2, chunk_ids.clone()).await?;
    println!("    ✅ Updated to version: {}", result.new_version);
    println!("    Change type: {:?}", result.change_summary.change_type);
    println!("    Similarity to previous: {:.2}", result.change_summary.similarity_to_previous);
    
    // Version 3: Major rewrite
    println!("  📄 Major rewrite...");
    let content_v3 = "Deep learning and neural networks represent the cutting edge of artificial intelligence. These systems can process vast amounts of data and learn complex patterns automatically.";
    
    let result = manager.update_document(doc_id, content_v3, chunk_ids).await?;
    println!("    ✅ Major update to version: {}", result.new_version);
    println!("    Change type: {:?}", result.change_summary.change_type);
    println!("    Similarity to previous: {:.2}", result.change_summary.similarity_to_previous);
    
    // Show version history
    println!("  📚 Version History:");
    if let Some(versions) = manager.get_version_history(doc_id) {
        for (i, version) in versions.iter().enumerate() {
            println!("    Version {}: {:?} at {}", 
                     version.version_number,
                     version.change_summary.change_type,
                     version.created_at.format("%H:%M:%S"));
        }
    }
    
    let stats = manager.get_stats();
    println!("  📊 Manager Statistics:");
    println!("    Total documents: {}", stats.total_documents);
    println!("    Total versions: {}", stats.total_versions);
    println!("    Unique content hashes: {}", stats.unique_content_hashes);
    
    Ok(())
}

// ================================================================================================
// TEST 4: DEDUPLICATION AND CONTENT FINGERPRINTING
// ================================================================================================

async fn test_deduplication() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Test 4: Deduplication and Content Fingerprinting");
    println!("-".repeat(50));
    
    // Test content fingerprinting
    println!("  🔍 Testing Content Fingerprinting...");
    
    let content1 = "The quick brown fox jumps over the lazy dog.";
    let content2 = "The quick brown fox jumps over the lazy dog."; // Identical
    let content3 = "The quick brown fox jumps over the sleepy dog."; // Minor change
    let content4 = "A completely different text about artificial intelligence."; // Major change
    
    let fp1 = ContentFingerprint::from_content(content1);
    let fp2 = ContentFingerprint::from_content(content2);
    let fp3 = ContentFingerprint::from_content(content3);
    let fp4 = ContentFingerprint::from_content(content4);
    
    println!("    Content 1 hash: {}", &fp1.content_hash[..8]);
    println!("    Content 2 hash: {}", &fp2.content_hash[..8]);
    println!("    Content 3 hash: {}", &fp3.content_hash[..8]);
    println!("    Content 4 hash: {}", &fp4.content_hash[..8]);
    
    println!("    Identical content detected: {}", fp1.content_hash == fp2.content_hash);
    println!("    Minor change detected: {}", fp1.content_hash != fp3.content_hash);
    println!("    Major change detected: {}", fp1.content_hash != fp4.content_hash);
    
    // Test structure preservation
    let markdown1 = "# Header\nSome content here";
    let markdown2 = "# Different Header\nOther content here";
    
    let md_fp1 = ContentFingerprint::from_content(markdown1);
    let md_fp2 = ContentFingerprint::from_content(markdown2);
    
    println!("    Structure preservation:");
    println!("      Same structure: {}", md_fp1.structure_hash == md_fp2.structure_hash);
    println!("      Different content: {}", md_fp1.content_hash != md_fp2.content_hash);
    
    // Test deduplication manager
    println!("  🗜️  Testing Deduplication Manager...");
    let mut manager = create_incremental_manager();
    
    // Add some documents with duplicate content
    let doc1 = Uuid::new_v4();
    let doc2 = Uuid::new_v4();
    let doc3 = Uuid::new_v4();
    
    let duplicate_content = "This content appears in multiple documents.";
    let unique_content = "This is unique content that won't be duplicated.";
    
    manager.update_document(doc1, duplicate_content, vec![Uuid::new_v4()]).await?;
    manager.update_document(doc2, duplicate_content, vec![Uuid::new_v4()]).await?;
    manager.update_document(doc3, unique_content, vec![Uuid::new_v4()]).await?;
    
    // Find duplicates
    let candidates = manager.find_duplicates_global().await?;
    println!("    Found {} deduplication candidates", candidates.len());
    
    let total_savings: usize = candidates.iter()
        .map(|c| c.potential_savings.bytes_saved)
        .sum();
    
    if total_savings > 0 {
        println!("    Potential storage savings: {} bytes", total_savings);
        
        // Apply deduplication
        let processed = manager.apply_deduplication(candidates).await?;
        println!("    Processed {} chunks for deduplication", processed.len());
    }
    
    let final_stats = manager.get_stats();
    println!("  📊 Final Deduplication Statistics:");
    println!("    Unique content hashes: {}", final_stats.unique_content_hashes);
    println!("    Duplicate content groups: {}", final_stats.duplicate_content_groups);
    
    Ok(())
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

#[allow(dead_code)]
fn print_chunk_details(chunk: &ai_interence_server::vector::TextChunk) {
    println!("  Chunk ID: {}", chunk.id);
    println!("  Index: {}", chunk.chunk_index);
    println!("  Type: {:?}", chunk.chunk_type);
    println!("  Tokens: {}", chunk.token_count);
    println!("  Position: {}-{}", chunk.start_position, chunk.end_position);
    
    if let Some(overlap) = &chunk.overlap_info {
        println!("  Overlap: {} tokens", overlap.overlap_tokens);
    }
    
    let preview = chunk.content.chars().take(100).collect::<String>();
    println!("  Content: {}...", preview);
}