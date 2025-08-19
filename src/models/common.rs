// ARCHITECTURE: Common Model Interface - Unified Model Abstraction
//
// DESIGN PHILOSOPHY:
// This module provides a unified interface for different AI models, enabling:
// 1. MODEL ABSTRACTION: Common interface for TinyLlama, Phi-3, and future models
// 2. POLYMORPHISM: Runtime model switching without code changes
// 3. CONSISTENCY: Standardized generation API across all models
// 4. EXTENSIBILITY: Easy addition of new models with minimal changes
//
// ENTERPRISE BENEFITS:
// - Zero-downtime model swapping between different architectures
// - Consistent performance metrics and monitoring across models
// - Simplified model management and deployment pipelines
// - A/B testing capabilities for model comparison

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use candle_core::Device;

/// Unified model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,           // Model identifier
    pub version: String,        // Model version
    pub parameters: usize,      // Total parameter count
    pub memory_mb: usize,       // Estimated memory usage
    pub device: String,         // Compute device
    pub vocab_size: usize,      // Tokenizer vocabulary size
    pub context_length: usize,  // Maximum sequence length
}

/// Unified model trait for all AI models
#[async_trait]
pub trait UnifiedModel: Send + Sync {
    /// Generate text from a prompt
    async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String>;
    
    /// Get model information
    fn model_info(&self) -> ModelInfo;
    
    /// Get the compute device being used
    fn device(&self) -> &Device;
    
    /// Get model name for identification
    fn name(&self) -> &str;
}

/// Enum wrapping different model implementations
pub enum ModelInstance {
    TinyLlama(crate::models::TinyLlamaModel),
    Phi3(crate::models::Phi3Model),
}

#[async_trait]
impl UnifiedModel for ModelInstance {
    async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        match self {
            ModelInstance::TinyLlama(model) => model.generate(prompt, max_tokens).await,
            ModelInstance::Phi3(model) => model.generate(prompt, max_tokens).await,
        }
    }
    
    fn model_info(&self) -> ModelInfo {
        match self {
            ModelInstance::TinyLlama(model) => model.model_info(),
            ModelInstance::Phi3(model) => model.model_info(),
        }
    }
    
    fn device(&self) -> &Device {
        match self {
            ModelInstance::TinyLlama(model) => &model.device,
            ModelInstance::Phi3(model) => &model.device,
        }
    }
    
    fn name(&self) -> &str {
        match self {
            ModelInstance::TinyLlama(_) => "TinyLlama-1.1B-Chat",
            ModelInstance::Phi3(_) => "Phi-3-mini-4k-instruct",
        }
    }
}

impl ModelInstance {
    /// Load a model by name
    pub async fn load_model(name: &str) -> Result<Self> {
        if name.contains("Phi-3") || name.contains("phi-3") {
            let phi3_model = crate::models::Phi3Model::load().await?;
            Ok(ModelInstance::Phi3(phi3_model))
        } else {
            let tinyllama_model = crate::models::TinyLlamaModel::load().await?;
            Ok(ModelInstance::TinyLlama(tinyllama_model))
        }
    }
}