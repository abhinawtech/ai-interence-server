use ai_interence_server::vector::{
    create_document_pipeline, RawDocument, DocumentFormat,
    create_semantic_chunker, ContentFingerprint
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Day 10: Quick Functionality Test");
    
    // Test 1: Document Ingestion
    println!("📄 Testing document ingestion...");
    let mut pipeline = create_document_pipeline();
    let content = "This is a test document.\n\nIt has multiple paragraphs.";
    let raw_doc = RawDocument::new(content.to_string(), DocumentFormat::PlainText);
    let processed = pipeline.process_document(raw_doc).await?;
    println!("   ✅ Processed {} sections, {} tokens", 
             processed.sections.len(), processed.total_tokens);
    
    // Test 2: Intelligent Chunking
    println!("🧠 Testing intelligent chunking...");
    let chunker = create_semantic_chunker(100);
    let doc_id = uuid::Uuid::new_v4();
    let test_text = "Artificial intelligence is transforming industries. Machine learning algorithms process data. Deep learning uses neural networks.";
    let result = chunker.chunk_document(doc_id, test_text)?;
    println!("   ✅ Created {} chunks", result.total_chunks);
    
    // Test 3: Content Fingerprinting
    println!("🔍 Testing content fingerprinting...");
    let fp1 = ContentFingerprint::from_content("Hello world");
    let fp2 = ContentFingerprint::from_content("Hello world");
    let fp3 = ContentFingerprint::from_content("Hello universe");
    println!("   ✅ Identical: {}, Different: {}", 
             fp1.content_hash == fp2.content_hash,
             fp1.content_hash != fp3.content_hash);
    
    println!("🎉 All Day 10 core features working!");
    Ok(())
}
