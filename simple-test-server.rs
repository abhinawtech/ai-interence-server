// Simple test server to verify Qdrant integration works
use axum::{
    extract::Query,
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simple server that tests Qdrant connectivity
    println!("🚀 Starting Simple Test Server...");
    
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/test-qdrant", get(test_qdrant))
        .route("/test-vector", get(test_vector_operation));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("🌐 Server running on http://{}", addr);
    println!("📡 Test endpoints:");
    println!("  • GET /health - Basic health check");
    println!("  • GET /test-qdrant - Test Qdrant connection");
    println!("  • GET /test-vector - Test vector operations");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "message": "Simple test server is running"
    }))
}

async fn test_qdrant() -> Result<Json<Value>, StatusCode> {
    println!("🧪 Testing Qdrant connection...");
    
    // Test Qdrant connection
    let client = reqwest::Client::new();
    let response = client
        .get("http://localhost:6333/collections")
        .send()
        .await;
    
    match response {
        Ok(resp) => {
            if resp.status().is_success() {
                let body: Value = resp.json().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                println!("✅ Qdrant connection successful");
                Ok(Json(json!({
                    "status": "success",
                    "message": "Qdrant is connected and responding",
                    "qdrant_response": body
                })))
            } else {
                println!("❌ Qdrant responded with error: {}", resp.status());
                Ok(Json(json!({
                    "status": "error",
                    "message": format!("Qdrant responded with status: {}", resp.status())
                })))
            }
        },
        Err(e) => {
            println!("❌ Failed to connect to Qdrant: {}", e);
            Ok(Json(json!({
                "status": "error",
                "message": format!("Failed to connect to Qdrant: {}", e)
            })))
        }
    }
}

async fn test_vector_operation() -> Result<Json<Value>, StatusCode> {
    println!("🧪 Testing vector operations...");
    
    let client = reqwest::Client::new();
    
    // Step 1: Create a test collection
    let collection_config = json!({
        "vectors": {
            "size": 5,
            "distance": "Cosine"
        }
    });
    
    let create_response = client
        .put("http://localhost:6333/collections/test_collection")
        .header("Content-Type", "application/json")
        .json(&collection_config)
        .send()
        .await;
    
    match create_response {
        Ok(resp) if resp.status().is_success() => {
            println!("✅ Collection created successfully");
            
            // Step 2: Insert a test vector
            let vector_data = json!({
                "points": [{
                    "id": 1,
                    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "payload": {
                        "title": "Test Vector",
                        "category": "test"
                    }
                }]
            });
            
            let insert_response = client
                .put("http://localhost:6333/collections/test_collection/points")
                .header("Content-Type", "application/json")
                .json(&vector_data)
                .send()
                .await;
            
            match insert_response {
                Ok(resp) if resp.status().is_success() => {
                    println!("✅ Vector inserted successfully");
                    
                    // Step 3: Search for similar vectors
                    let search_data = json!({
                        "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "limit": 5,
                        "with_payload": true
                    });
                    
                    let search_response = client
                        .post("http://localhost:6333/collections/test_collection/points/search")
                        .header("Content-Type", "application/json")
                        .json(&search_data)
                        .send()
                        .await;
                    
                    match search_response {
                        Ok(resp) if resp.status().is_success() => {
                            let search_result: Value = resp.json().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                            println!("✅ Vector search successful");
                            
                            Ok(Json(json!({
                                "status": "success",
                                "message": "All vector operations completed successfully",
                                "operations": {
                                    "collection_created": true,
                                    "vector_inserted": true,
                                    "search_completed": true
                                },
                                "search_results": search_result
                            })))
                        },
                        Ok(resp) => {
                            println!("❌ Search failed with status: {}", resp.status());
                            Ok(Json(json!({
                                "status": "partial_success",
                                "message": "Collection and insertion succeeded, but search failed",
                                "search_error": resp.status().to_string()
                            })))
                        },
                        Err(e) => {
                            println!("❌ Search request failed: {}", e);
                            Ok(Json(json!({
                                "status": "partial_success", 
                                "message": "Collection and insertion succeeded, but search request failed",
                                "search_error": e.to_string()
                            })))
                        }
                    }
                },
                Ok(resp) => {
                    println!("❌ Vector insertion failed with status: {}", resp.status());
                    Ok(Json(json!({
                        "status": "partial_success",
                        "message": "Collection created but vector insertion failed",
                        "insert_error": resp.status().to_string()
                    })))
                },
                Err(e) => {
                    println!("❌ Vector insertion request failed: {}", e);
                    Ok(Json(json!({
                        "status": "partial_success",
                        "message": "Collection created but vector insertion request failed",
                        "insert_error": e.to_string()
                    })))
                }
            }
        },
        Ok(resp) => {
            println!("❌ Collection creation failed with status: {}", resp.status());
            Ok(Json(json!({
                "status": "error",
                "message": format!("Failed to create collection: {}", resp.status())
            })))
        },
        Err(e) => {
            println!("❌ Collection creation request failed: {}", e);
            Ok(Json(json!({
                "status": "error",
                "message": format!("Failed to create collection: {}", e)
            })))
        }
    }
}