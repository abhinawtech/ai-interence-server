pub mod llama;
pub mod llama_generic;
pub mod codellama;
pub mod deepseek_coder;
pub mod version_manager;
pub mod failover_manager;
pub mod phi3;
pub mod gemma;
pub mod gemma_poc;
pub mod atomic_swap;
pub mod traits;
pub mod registry;

pub use llama::{TinyLlamaModel};
pub use llama_generic::GenericLlamaModel;
pub use codellama::CodeLlamaModel;
pub use deepseek_coder::DeepSeekCoderModel;
pub use phi3::Phi3Model;
pub use gemma_poc::GemmaModel;
pub use version_manager::{ModelVersionManager, ModelVersion, HealthCheckResult};
pub use failover_manager::AutomaticFailoverManager;
pub use atomic_swap::{AtomicModelSwap, SwapResult, SwapSafetyReport};
pub use traits::{ModelTrait, ModelInfo, BoxedModel, ModelHealthCheck};
pub use registry::{ModelRegistry, global_registry, initialize_registry};

use anyhow::Result;

// New trait-based ModelInstance using dynamic dispatch
pub struct ModelInstance {
    inner: BoxedModel,
}

impl std::fmt::Debug for ModelInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ModelInstance({})", self.inner.model_name())
    }
}

impl ModelInstance {
    /// Create a ModelInstance from a boxed model trait
    pub fn new(model: BoxedModel) -> Self {
        Self { inner: model }
    }

    /// Generate text using the underlying model
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.inner.generate(prompt, max_tokens).await
    }

    /// Get the compute device
    pub fn device(&self) -> &candle_core::Device {
        self.inner.device()
    }

    /// Load a model by name using the registry
    pub async fn load_by_name(name: &str) -> Result<Self> {
        let model = global_registry().load_model(name).await?;
        Ok(Self::new(model))
    }

    /// Get model information
    pub fn model_info(&self) -> ModelInfo {
        self.inner.model_info()
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    /// Perform health check
    pub async fn health_check(&mut self) -> Result<ModelHealthCheck> {
        self.inner.health_check().await
    }

    /// Check if model supports a feature
    pub fn supports_feature(&self, feature: &str) -> bool {
        self.inner.supports_feature(feature)
    }
}

/// Initialize all models in the registry
/// This should be called at application startup
pub fn initialize_models() -> Result<()> {
    tracing::info!("🚀 Initializing model registry with available models...");
    
    // Initialize the registry
    initialize_registry();
    
    // Register all available models
    if let Err(e) = gemma_poc::register_gemma_model() {
        tracing::warn!("Failed to register Gemma model: {}", e);
    }
    
    // Enable TinyLlama for fast inference testing
    if let Err(e) = llama::register_tinyllama_model() {
        tracing::warn!("Failed to register TinyLlama model: {}", e);
    }
    
    // Register CodeLlama for code generation tasks
    if let Err(e) = codellama::register_codellama_model() {
        tracing::warn!("Failed to register CodeLlama model: {}", e);
    }
    
    // Register Generic Llama for flexible model loading
    if let Err(e) = llama_generic::register_generic_llama_model() {
        tracing::warn!("Failed to register Generic Llama model: {}", e);
    }
    
    // Register DeepSeek Coder for specialized code generation
    if let Err(e) = deepseek_coder::register_deepseek_coder_model() {
        tracing::warn!("Failed to register DeepSeek Coder model: {}", e);
    }
    // if let Err(e) = phi3::register_phi3_model() {
    //     tracing::warn!("Failed to register Phi3 model: {}", e);
    // }
    
    let registry = global_registry();
    let models = registry.list_models();
    tracing::info!("✅ Registered {} models in registry:", models.len());
    
    for model in models {
        tracing::info!("  📋 {} ({}): {} aliases", 
                      model.name, 
                      model.model_type,
                      model.aliases.len());
    }
    
    Ok(())
}
