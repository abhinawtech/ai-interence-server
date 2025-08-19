// EXAMPLE: How to add a new LLM to the extensible architecture
// This demonstrates how simple it is to add any new model

use anyhow::Result;
use async_trait::async_trait;
use candle_core::Device;
use crate::models::traits::{ModelTrait, BoxedModel, ModelInfo, ModelHealthCheck};
use crate::models::registry::{global_registry, ModelRegistration};
use std::sync::Arc;

/// Example new model - could be LLaMA, Claude, GPT, Mistral, etc.
#[derive(Debug)]
pub struct ExampleNewModel {
    pub device: Device,
    model_name: String,
}

impl ExampleNewModel {
    pub async fn load() -> Result<Self> {
        // Your model loading logic here
        let device = Device::Cpu; // or GPU detection logic
        
        Ok(Self {
            device,
            model_name: "example-new-model".to_string(),
        })
    }
}

// Step 1: Implement ModelTrait (the only requirement!)
#[async_trait]
impl ModelTrait for ExampleNewModel {
    async fn load() -> Result<BoxedModel>
    where
        Self: Sized,
    {
        let model = Self::load().await?;
        Ok(Box::new(model))
    }

    async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Your generation logic here
        Ok(format!("Generated response to: {} (max_tokens: {})", prompt, max_tokens))
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "Example New Model".to_string(),
            version: "v1.0".to_string(),
            parameters: 7_000_000_000, // 7B parameters
            memory_mb: 14000,
            device: format!("{:?}", self.device),
            vocab_size: 32000,
            context_length: 4096,
            model_type: "example".to_string(),
            architecture: "transformer".to_string(),
            precision: "f16".to_string(),
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "generation" => true,
            "health_check" => true,
            "streaming" => false, // example: doesn't support streaming
            _ => false,
        }
    }
}

// Step 2: Create registration function (one-time setup)
pub fn register_example_model() -> Result<()> {
    async fn example_factory() -> Result<BoxedModel> {
        let model = ExampleNewModel::load().await?;
        Ok(Box::new(model))
    }

    let registration = ModelRegistration {
        name: "example-new-model".to_string(),
        aliases: vec![
            "example".to_string(),
            "new-model".to_string(),
            "demo".to_string(),
        ],
        description: "Example new model showing extensibility".to_string(),
        model_type: "example".to_string(),
        supported_features: vec![
            "generation".to_string(),
            "health_check".to_string(),
        ],
        memory_requirements_mb: 14000,
        factory: Arc::new(Box::new(|| Box::new(Box::pin(example_factory())))),
    };

    global_registry().register_model(registration)?;
    Ok(())
}

// Step 3: Add one line to initialize_models() in src/models/mod.rs:
// ```rust
// if let Err(e) = example_new_model::register_example_model() {
//     tracing::warn!("Failed to register Example model: {}", e);
// }
// ```

// That's it! Your new model is now available via:
// - "example-new-model" (canonical name)  
// - "example", "new-model", "demo" (aliases)
// 
// Usage:
// curl -X POST http://localhost:3000/api/v1/generate \
//   -H "Content-Type: application/json" \
//   -d '{"model": "example", "prompt": "Hello!", "max_tokens": 10}'

/*
KEY BENEFITS OF THE NEW ARCHITECTURE:

1. ✅ ZERO CORE CHANGES: No modifications to server, API, or batch processing code
2. ✅ SELF-CONTAINED: Each model is in its own module  
3. ✅ MULTIPLE ALIASES: Support different naming conventions
4. ✅ FEATURE FLAGS: Models can advertise their capabilities
5. ✅ RUNTIME DISCOVERY: Models register themselves at startup
6. ✅ TYPE SAFETY: Trait system ensures consistent interface
7. ✅ ASYNC READY: Full async/await support throughout

COMPARISON:
- OLD: Adding a model required changing 4+ files, recompiling entire codebase
- NEW: Adding a model requires 1 new file + 1 line registration call

This enables:
- Plugin-style model distribution
- Easy A/B testing of different models  
- Customer-specific model deployments
- Rapid prototyping and experimentation
*/