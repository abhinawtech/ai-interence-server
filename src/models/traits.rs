// ARCHITECTURE: Trait-Based LLM Abstraction Layer
//
// DESIGN PHILOSOPHY:
// This module implements a trait-based architecture that enables:
// 1. RUNTIME POLYMORPHISM: Support for multiple LLM types without compile-time coupling
// 2. EXTENSIBILITY: Easy addition of new models without core code changes  
// 3. UNIFORM INTERFACE: Consistent API across all model implementations
// 4. PLUGIN ARCHITECTURE: Models can be loaded dynamically at runtime
//
// BENEFITS:
// - Add new LLMs by implementing the ModelTrait
// - No more hardcoded enums or match statements
// - Clean separation of model-specific logic
// - Future-proof for new model architectures

use anyhow::Result;
use async_trait::async_trait;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Core model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub parameters: usize,
    pub memory_mb: usize,
    pub device: String,
    pub vocab_size: usize,
    pub context_length: usize,
    pub model_type: String,        // "llama", "gemma", "phi3", etc.
    pub architecture: String,      // "transformer", "mamba", etc.
    pub precision: String,         // "f16", "f32", "int8", etc.
}

/// Health check result for model validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealthCheck {
    pub status: bool,
    pub score: f32,
    pub latency_ms: u64,
    pub message: String,
    pub generation_test: Option<String>,
}

/// Unified trait interface for all LLM implementations
/// 
/// This trait defines the contract that all language models must implement,
/// enabling runtime polymorphism and easy extensibility
#[async_trait]
pub trait ModelTrait: Send + Sync + Debug {
    /// Load the model asynchronously
    /// This should handle all initialization including:
    /// - Weight loading
    /// - Tokenizer setup  
    /// - Device selection
    /// - Configuration parsing
    async fn load() -> Result<Box<dyn ModelTrait>>
    where
        Self: Sized;

    /// Generate text from a prompt
    /// 
    /// # Arguments
    /// * `prompt` - Input text to generate from
    /// * `max_tokens` - Maximum number of tokens to generate
    /// 
    /// # Returns
    /// Generated text as a String
    async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String>;

    /// Get model information and metadata
    fn model_info(&self) -> ModelInfo;

    /// Get the compute device being used
    fn device(&self) -> &Device;

    /// Perform a health check on the model
    /// Default implementation does basic generation test
    async fn health_check(&mut self) -> Result<ModelHealthCheck> {
        let start = std::time::Instant::now();
        
        match self.generate("Hello", 3).await {
            Ok(response) => {
                let latency = start.elapsed().as_millis() as u64;
                Ok(ModelHealthCheck {
                    status: true,
                    score: 1.0,
                    latency_ms: latency,
                    message: "Health check passed".to_string(),
                    generation_test: Some(response),
                })
            }
            Err(e) => {
                let latency = start.elapsed().as_millis() as u64;
                Ok(ModelHealthCheck {
                    status: false,
                    score: 0.0,
                    latency_ms: latency,
                    message: format!("Health check failed: {}", e),
                    generation_test: None,
                })
            }
        }
    }

    /// Get the model's unique identifier/name
    fn model_name(&self) -> &str;

    /// Check if the model supports a specific feature
    /// Examples: "batching", "streaming", "fine_tuning", etc.
    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "generation" => true,
            "health_check" => true,
            _ => false,
        }
    }

    /// Get model-specific configuration as JSON
    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "model_info": self.model_info(),
            "supported_features": [
                "generation",
                "health_check"
            ]
        })
    }
}

/// Type alias for boxed model trait objects
pub type BoxedModel = Box<dyn ModelTrait>;

/// Model factory function signature
/// Each model implementation should provide a factory function matching this signature
pub type ModelFactory = fn() -> Box<dyn std::future::Future<Output = Result<BoxedModel>> + Send + Unpin>;

/// Helper trait for cloning trait objects (if needed in future)
pub trait CloneableModel: ModelTrait {
    fn clone_box(&self) -> BoxedModel;
}