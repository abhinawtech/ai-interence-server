// INTEGRATION: Working PoC Gemma Model - Direct Integration from Standalone Version
// This implementation directly integrates the proven working PoC code
// Key difference: Creates fresh model instance per request to avoid shared state corruption

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma::{Config, Model as Gemma};
use hf_hub::{Repo, RepoType, api::tokio::{Api, ApiBuilder}};
use std::fs;
use tokenizers::Tokenizer;
use crate::models::traits::{ModelTrait, ModelInfo, BoxedModel, ModelHealthCheck};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Gemma Model Implementation - Direct PoC Integration with Fresh Instance Strategy  
#[derive(Debug)]
pub struct GemmaModel {
    tokenizer: Tokenizer,           // Tokenizer (safe to share)
    pub device: Device,             // Compute device 
    config: Config,                 // Model architecture configuration
    weight_files: Vec<std::path::PathBuf>, // Cached weight files to avoid re-downloading
    // CORRECTNESS FIX: Use fresh model instances per request to avoid tensor shape corruption
    // This matches the working standalone implementation strategy
}

impl GemmaModel {
    pub async fn load() -> Result<Self> {
        tracing::info!("🚀 Starting Gemma-2B model loading (PoC Integration)...");

        // GPU Acceleration Strategy - same as your PoC
        let device = if candle_core::utils::metal_is_available() {
            tracing::info!("🚀 Metal GPU available, using acceleration");
            match Device::new_metal(0) {
                Ok(d) => {
                    tracing::info!("✅ Successfully initialized Metal device");
                    d
                }
                Err(e) => {
                    tracing::warn!("❌ Failed to initialize Metal: {}, falling back to CPU", e);
                    Device::Cpu
                }
            }
        } else if candle_core::utils::cuda_is_available() {
            tracing::info!("🚀 CUDA GPU available, using acceleration");
            Device::new_cuda(0).unwrap_or_else(|e| {
                tracing::warn!("❌ Failed to initialize CUDA: {}, using CPU", e);
                Device::Cpu
            })
        } else {
            tracing::info!("🖥️  Using CPU for inference");
            Device::Cpu
        };
        
        tracing::info!("📱 Final selected device: {:?}", device);

        // Load model components using your PoC approach
        let (tokenizer, config, weight_files) = Self::load_model_components(&device).await?;

        tracing::info!("✅ Gemma-2B model loaded successfully with fresh instance strategy (PoC approach)");
        Ok(Self {
            tokenizer,
            device,
            config,
            weight_files,
        })
    }

    async fn load_model_components(device: &Device) -> Result<(Tokenizer, Config, Vec<std::path::PathBuf>)> {
        // EXACT COPY of your working PoC loading logic
        let api = if let Ok(token) = std::env::var("HF_TOKEN") {
            tracing::info!("🔑 Using HuggingFace token for authentication");
            ApiBuilder::new()
                .with_token(Some(token))
                .build()?
        } else {
            tracing::warn!("⚠️ No HF_TOKEN found, using anonymous access");
            Api::new()?
        };
        
        let repo = api.repo(Repo::with_revision(
            "google/gemma-2b-it".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        tracing::info!("📥 Downloading tokenizer...");
        let tokenizer_filename = repo.get("tokenizer.json").await?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        tracing::info!("✅ Tokenizer loaded successfully");

        tracing::info!("📥 Downloading model config...");
        let config_file = repo.get("config.json").await
            .map_err(|e| anyhow::anyhow!("Failed to download config.json: {}", e))?;
        let config_content = std::fs::read_to_string(config_file)?;
        let config: Config = serde_json::from_str(&config_content)?;
        tracing::info!("✅ Config loaded successfully");

        // Download weight files once during initialization
        tracing::info!("📥 Downloading model weights...");
        let weight_files = vec![
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        
        let mut downloaded_files = Vec::new();
        for file in weight_files {
            match repo.get(file).await {
                Ok(path) => {
                    tracing::info!("✅ Downloaded: {}", file);
                    downloaded_files.push(path);
                }
                Err(e) => {
                    tracing::warn!("⚠️  Couldn't download {}: {}", file, e);
                    // Try single file fallback
                    if downloaded_files.is_empty() {
                        if let Ok(single_file) = repo.get("model.safetensors").await {
                            tracing::info!("✅ Using single model file");
                            downloaded_files.push(single_file);
                            break;
                        }
                    }
                }
            }
        }
        
        if downloaded_files.is_empty() {
            return Err(anyhow::anyhow!("No model weight files could be downloaded"));
        }

        Ok((tokenizer, config, downloaded_files))
    }

    // Create model instance - used once during initialization  
    async fn create_model_instance(device: &Device, config: &Config, weight_files: &[std::path::PathBuf]) -> Result<Gemma> {
        tracing::debug!("🔄 Creating Gemma model instance");
        
        // Use cached weight files from initialization
        tracing::debug!("🔧 Loading model weights from cached files...");
        
        // COMPATIBILITY: Device-specific precision selection (your PoC + our fix)
        let dtype = match device {
            Device::Cpu => {
                tracing::debug!("🖥️  Using F32 precision on CPU (accelerate backend limitation)");
                DType::F32
            }
            Device::Metal(_) | Device::Cuda(_) => {
                tracing::debug!("🚀 Using F16 precision on GPU for memory efficiency");
                DType::F16
            }
        };
        
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(weight_files, dtype, device)? 
        };
        tracing::debug!("✅ Model weights loaded from cache");

        tracing::debug!("🏗️  Building Gemma model...");
        let model = Gemma::new(false, config, vb)?;
        tracing::debug!("✅ Gemma model built successfully");

        Ok(model)
    }

    // CORRECTNESS FIX: Use fresh model instances per request (matches working standalone code)
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        tracing::debug!("🚀 Gemma generating response for: '{}' (max_tokens: {})", prompt, max_tokens);
        
        // EXACT COPY of your working PoC generation logic
        let tokens = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let token_ids = tokens.get_ids().iter().map(|&id| id as u32).collect::<Vec<_>>();
        tracing::debug!("✅ Input tokenized: {} tokens", token_ids.len());

        // CORRECTNESS FIX: Create fresh model instance per request to avoid tensor shape corruption
        // This matches your working standalone implementation exactly
        tracing::debug!("🔄 Creating fresh Gemma model instance for generation");
        let mut model = Self::create_model_instance(&self.device, &self.config, &self.weight_files).await?;
        let generated_text = self.generate_text(&mut model, &token_ids, max_tokens).await?;

        tracing::debug!("✅ Gemma generation completed successfully");
        Ok(generated_text)
    }

    // EXACT COPY of your working PoC generate_text function
    async fn generate_text(&self, model: &mut Gemma, input_ids: &[u32], max_new_tokens: usize) -> Result<String> {
        let mut generated_tokens = input_ids.to_vec();
        let max_seq_len = 512; // Limit sequence length to avoid memory issues
        
        for step in 0..max_new_tokens {
            // Ensure sequence doesn't get too long
            if generated_tokens.len() >= max_seq_len {
                tracing::debug!("⚠️  Reached maximum sequence length, stopping generation");
                break;
            }
            
            // EXACT COPY: For first step, use full prompt; for subsequent steps, use only the last token
            let (input_tensor, seq_offset) = if step == 0 {
                // First forward pass with full prompt
                let input_tensor = Tensor::from_vec(
                    generated_tokens.clone(),
                    (1, generated_tokens.len()),
                    &self.device,
                )?;
                (input_tensor, 0)
            } else {
                // Subsequent passes with just the last token
                let last_token = generated_tokens[generated_tokens.len() - 1];
                let input_tensor = Tensor::from_vec(
                    vec![last_token],
                    (1, 1),
                    &self.device,
                )?;
                (input_tensor, generated_tokens.len() - 1)
            };
            
            // Forward pass
            let logits = model.forward(&input_tensor, seq_offset)?;
            
            // Get the logits shape for debugging
            let logits_shape = logits.dims();
            if step == 0 {
                tracing::debug!("🔍 Logits shape: {:?}", logits_shape);
            }
            
            // EXACT COPY: For Gemma, logits shape is [batch_size, seq_len, vocab_size]
            let last_logits = if logits_shape[1] == 1 {
                // Model returns logits for just the last position
                logits.i((0, 0))?
            } else {
                // Model returns logits for all positions, get the last one
                let last_pos = generated_tokens.len() - 1;
                if last_pos >= logits_shape[1] {
                    tracing::debug!("❌ Position {} exceeds logits sequence length {}", last_pos, logits_shape[1]);
                    break;
                }
                logits.i((0, last_pos))?
            };
            
            // Sample next token (greedy sampling)
            let next_token = last_logits.argmax(candle_core::D::Minus1)?;
            let next_token_id = next_token.to_scalar::<u32>()?;
            
            // Debug: Print token ID and decoded text for each step
            tracing::debug!("🎯 Step {}: token ID {} -> '{}'", 
                    step, 
                    next_token_id, 
                    self.tokenizer.decode(&[next_token_id], false).unwrap_or_else(|_| "?".to_string()));
            
            // Check for EOS tokens
            if self.is_eos_token(next_token_id) {
                tracing::debug!("📄 Hit EOS token, stopping generation");
                break;
            }
            
            generated_tokens.push(next_token_id);
        }
        
        // Decode only the newly generated tokens
        let new_tokens = &generated_tokens[input_ids.len()..];
        tracing::debug!("🔤 Decoding {} new tokens", new_tokens.len());
        
        let generated_text = self.tokenizer
            .decode(new_tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode generated tokens: {}", e))?;
        
        Ok(generated_text)
    }

    // EXACT COPY of your working PoC is_eos_token function
    fn is_eos_token(&self, token_id: u32) -> bool {
        // Common EOS tokens for Gemma
        if let Some(eos_id) = self.tokenizer.token_to_id("<end_of_turn>") {
            if token_id == eos_id {
                return true;
            }
        }
        if let Some(eos_id) = self.tokenizer.token_to_id("</s>") {
            if token_id == eos_id {
                return true;
            }
        }
        if let Some(eos_id) = self.tokenizer.token_to_id("<|endoftext|>") {
            if token_id == eos_id {
                return true;
            }
        }
        
        // Common EOS token IDs as fallback
        matches!(token_id, 1 | 2 | 106 | 107)
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "Gemma-2B-IT-PoC".to_string(),
            version: "main".to_string(),
            parameters: self.estimate_parameters(),
            memory_mb: self.estimate_memory_usage(),
            device: format!("{:?}", self.device),
            vocab_size: self.config.vocab_size,
            context_length: self.config.max_position_embeddings,
            model_type: "gemma".to_string(),
            architecture: "transformer".to_string(),
            precision: "f16".to_string(),
        }
    }

    fn estimate_parameters(&self) -> usize {
        2_000_000_000 // 2 billion parameters
    }

    fn estimate_memory_usage(&self) -> usize {
        let params = self.estimate_parameters();
        let param_memory = params * 2; // FP16 precision
        let activation_memory = 8 * 1024 * 1024; // 8MB estimate for activations
        (param_memory + activation_memory) / (1024 * 1024)
    }
}

// Implement ModelTrait for GemmaModel to support the new architecture
#[async_trait]
impl ModelTrait for GemmaModel {
    async fn load() -> Result<BoxedModel>
    where
        Self: Sized,
    {
        let model = Self::load().await?;
        Ok(Box::new(model))
    }

    async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate(prompt, max_tokens).await
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "Gemma-2B-IT-PoC".to_string(),
            version: "main".to_string(),
            parameters: self.estimate_parameters(),
            memory_mb: self.estimate_memory_usage(),
            device: format!("{:?}", self.device),
            vocab_size: self.config.vocab_size,
            context_length: self.config.max_position_embeddings,
            model_type: "gemma".to_string(),
            architecture: "transformer".to_string(),
            precision: "f16".to_string(),
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn model_name(&self) -> &str {
        "gemma-2b-it"
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "generation" => true,
            "health_check" => true,
            "fresh_instances" => true,
            "metal_gpu" => true,
            _ => false,
        }
    }
}

// Register the Gemma model with the registry
pub fn register_gemma_model() -> Result<()> {
    use crate::models::registry::{global_registry, ModelRegistration};
    use std::sync::Arc;

    async fn gemma_factory() -> Result<BoxedModel> {
        let model = GemmaModel::load().await?;
        Ok(Box::new(model))
    }

    let registration = ModelRegistration {
        name: "gemma-2b-it".to_string(),
        aliases: vec![
            "gemma".to_string(),
            "gemma-2b".to_string(),
            "gemma2b".to_string(),
        ],
        description: "Google Gemma 2B Instruction-Tuned model with fresh instance strategy (matches working standalone)".to_string(),
        model_type: "gemma".to_string(),
        supported_features: vec![
            "generation".to_string(),
            "health_check".to_string(),
            "fresh_instances".to_string(),
            "metal_gpu".to_string(),
        ],
        memory_requirements_mb: 4500,
        factory: Arc::new(Box::new(|| Box::new(Box::pin(gemma_factory())))),
    };

    global_registry().register_model(registration)?;
    Ok(())
}