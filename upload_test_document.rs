// Document Upload Utility for Day 10 Testing
use reqwest::Client;
use serde_json::json;
use std::env;
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("📁 Document Upload Utility for Day 10");
        println!("Usage: {} <file_path>", args[0]);
        println!("       {} --demo", args[0]);
        return Ok(());
    }

    let client = Client::new();
    let base_url = "http://localhost:3000";

    if args[1] == "--demo" {
        create_demo_documents().await?;
        return Ok(());
    }

    let file_path = &args[1];
    upload_document(&client, base_url, file_path).await?;
    
    Ok(())
}

async fn upload_document(client: &Client, base_url: &str, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(file_path);
    
    if !path.exists() {
        println!("❌ File not found: {}", file_path);
        return Ok(());
    }

    println!("📖 Reading file: {}", file_path);
    let content = fs::read_to_string(path)?;
    let format = detect_format(path);
    
    println!("🔍 Format: {:?}, Size: {} bytes", format, content.len());

    let payload = json!({
        "content": content,
        "format": format,
        "source_path": file_path
    });

    let response = client
        .post(&format!("{}/api/v1/documents/ingest", base_url))
        .json(&payload)
        .send()
        .await?;

    if response.status().is_success() {
        let result: serde_json::Value = response.json().await?;
        println!("✅ Document ingested successfully!");
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("❌ Upload failed: {}", response.text().await?);
    }

    Ok(())
}

async fn create_demo_documents() -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Creating demo documents...");
    
    let demo_md = r#"# Machine Learning Guide

## Introduction
Machine learning is a subset of artificial intelligence.

## Key Concepts
- Supervised Learning
- Unsupervised Learning  
- Deep Learning

## Applications
1. Image recognition
2. Natural language processing
3. Recommendation systems
"#;

    fs::write("demo_ml_guide.md", demo_md)?;
    println!("✅ Created demo_ml_guide.md");

    let demo_json = r#"{"model": "ai-assistant", "version": "1.0", "features": ["text_generation"]}"#;
    fs::write("demo_config.json", demo_json)?;
    println!("✅ Created demo_config.json");

    let demo_csv = "name,age,role\nAlice,28,Engineer\nBob,32,Manager";
    fs::write("demo_data.csv", demo_csv)?;
    println!("✅ Created demo_data.csv");

    println!("\n🎯 Test with: cargo run --bin upload_test_document demo_ml_guide.md");
    Ok(())
}

fn detect_format(path: &Path) -> String {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("md") => "Markdown",
        Some("json") => "Json", 
        Some("csv") => "Csv",
        _ => "PlainText",
    }.to_string()
}
