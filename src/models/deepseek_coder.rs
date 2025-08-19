// ARCHITECTURE: DeepSeek Coder 1.3B Model Implementation - Specialized Code Generation Engine
//
// DESIGN PHILOSOPHY:
// DeepSeek Coder is a code-specialized language model optimized for:
// 1. CODE GENERATION: Superior performance on programming tasks vs general models
// 2. INSTRUCTION FOLLOWING: Fine-tuned for code completion, debugging, and explanation
// 3. MULTI-LANGUAGE: Supports 80+ programming languages with strong performance
// 4. EFFICIENCY: 1.3B parameters provide good balance between capability and speed
//
// PERFORMANCE CHARACTERISTICS:
// - Model Size: 1.3B parameters (~2.6GB in F16 format)
// - Memory Usage: ~3.2GB total (weights + activations)
// - Inference Speed: 8-15 tokens/second on Apple Silicon (faster than CodeLlama-7B)
// - Context Length: 16K tokens (4x larger than TinyLlama)
// - Specialization: Code completion, debugging, explanation, documentation

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama};
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde_json::Value;
use std::fs;
use tokenizers::Tokenizer;
use std::sync::Arc;
use crate::models::traits::{ModelTrait, ModelInfo, BoxedModel};
use async_trait::async_trait;

/// DeepSeek Coder 1.3B Model - Specialized Code Generation Engine
#[derive(Debug)]
pub struct DeepSeekCoderModel {
    model: Arc<Llama>,           // Thread-safe model reference
    tokenizer: Arc<Tokenizer>,   // Code-aware tokenizer
    pub device: Device,          // Compute device (GPU preferred for better speed)
    config: Config,              // Model architecture configuration
}

impl DeepSeekCoderModel {
    pub async fn load() -> Result<Self> {
        tracing::info!("🚀 Starting DeepSeek Coder 1.3B model loading...");

        // GPU ACCELERATION: Recommended for better performance
        // 1.3B models benefit significantly from GPU acceleration
        let device = if candle_core::utils::metal_is_available() {
            tracing::info!("🚀 Metal GPU available, using acceleration (recommended for DeepSeek Coder)");
            match Device::new_metal(0) {
                Ok(d) => {
                    tracing::info!("✅ Successfully initialized Metal device for DeepSeek Coder");
                    d
                }
                Err(e) => {
                    tracing::warn!("❌ Failed to initialize Metal: {}, falling back to CPU (performance will be slower)", e);
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
            tracing::warn!("🖥️  Using CPU for DeepSeek Coder 1.3B - expect slower performance (~3-5 tok/s)");
            Device::Cpu
        };
        
        tracing::info!("📱 DeepSeek Coder device: {:?}", device);

        // Load DeepSeek Coder components
        let (model, tokenizer, config) = Self::load_model_components(&device).await?;

        tracing::info!("✅ DeepSeek Coder 1.3B model loaded successfully");
        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            device,
            config,
        })
    }

    async fn load_model_components(device: &Device) -> Result<(Llama, Tokenizer, Config)> {
        let device_clone = device.clone();
        let (model, tokenizer, config) =
            tokio::task::spawn_blocking(move || Self::load_model_sync(device_clone)).await??;
        Ok((model, tokenizer, config))
    }

    fn load_model_sync(device: Device) -> Result<(Llama, Tokenizer, Config)> {
        // Try to load without authentication first, fall back to authenticated if needed
        let api = if let Ok(token) = std::env::var("HF_TOKEN") {
            tracing::info!("🔑 Using HuggingFace token for DeepSeek Coder authentication");
            hf_hub::api::sync::ApiBuilder::new()
                .with_token(Some(token))
                .build()?
        } else {
            tracing::info!("📖 Attempting anonymous access for DeepSeek Coder");
            Api::new()?
        };
        
        let repo = api.repo(Repo::with_revision(
            "deepseek-ai/deepseek-coder-1.3b-instruct".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        tracing::info!("📥 Downloading DeepSeek Coder configuration files...");
        let config_file = repo
            .get("config.json")
            .map_err(|e| anyhow::anyhow!("Failed to download DeepSeek Coder config.json: {}", e))?;

        let tokenizer_file = repo
            .get("tokenizer.json")
            .map_err(|e| anyhow::anyhow!("Failed to download DeepSeek Coder tokenizer.json: {}", e))?;

        let config_json = fs::read_to_string(&config_file)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek Coder tokenizer: {}", e))?;

        let config = Self::parse_deepseek_config(&config_json)?;
        tracing::info!(
            "📊 DeepSeek Coder Config: vocab_size={}, hidden_size={}, num_layers={}, context_length={}",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.max_position_embeddings
        );

        tracing::info!("📥 Downloading DeepSeek Coder 1.3B weights (~2.6GB)...");
        let weight_files = Self::download_deepseek_weights(&repo)?;
        let vars = Self::load_weights(&weight_files, &device)?;

        tracing::info!("🏗️  Building DeepSeek Coder 1.3B model graph...");
        let model = Llama::load(vars, &config)?;

        tracing::info!("✅ DeepSeek Coder 1.3B model components loaded successfully");
        Ok((model, tokenizer, config))
    }

    fn parse_deepseek_config(config_json: &str) -> Result<Config> {
        let config: Value = serde_json::from_str(config_json)?;

        // DeepSeek Coder 1.3B specific parameters
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(2048) as usize;
        let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(5632) as usize;
        let num_hidden_layers = config["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
        let num_attention_heads = config["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        let num_key_value_heads = config
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(16); // DeepSeek Coder uses full attention heads
        let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let rope_theta = config
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let max_position_embeddings = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(16384) as usize; // 16K context length

        Ok(Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps,
            rope_theta: rope_theta as f32,
            max_position_embeddings,
            bos_token_id: Some(
                config
                    .get("bos_token_id")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(1) as u32,
            ),
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(
                config
                    .get("eos_token_id")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(2) as u32,
            )),
            rope_scaling: None,
            tie_word_embeddings: config
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            use_flash_attn: false,
        })
    }

    fn download_deepseek_weights(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<std::path::PathBuf>> {
        // DeepSeek Coder 1.3B weight file patterns
        let possible_patterns = vec![
            // SafeTensors (preferred)
            (1..=3)
                .map(|i| format!("model-{i:05}-of-00003.safetensors"))
                .collect::<Vec<_>>(),
            vec!["model.safetensors".to_string()],
            // PyTorch fallback
            (1..=2)
                .map(|i| format!("pytorch_model-{i:05}-of-00002.bin"))
                .collect::<Vec<_>>(),
            vec!["pytorch_model.bin".to_string()],
        ];

        for pattern in possible_patterns {
            let mut pattern_files = Vec::new();
            let mut all_found = true;

            for filename in &pattern {
                match repo.get(filename) {
                    Ok(path) => {
                        tracing::info!("✅ Downloaded DeepSeek Coder weight file: {}", filename);
                        pattern_files.push(path);
                    }
                    Err(_) => {
                        all_found = false;
                        break;
                    }
                }
            }

            if all_found && !pattern_files.is_empty() {
                tracing::info!("✅ Successfully downloaded {} DeepSeek Coder weight file(s)", pattern_files.len());
                return Ok(pattern_files);
            }
        }

        Err(anyhow::anyhow!("No DeepSeek Coder weight files found"))
    }

    fn load_weights<'a>(
        weight_files: &'a [std::path::PathBuf],
        device: &'a Device,
    ) -> Result<VarBuilder<'a>> {
        if weight_files[0].extension().and_then(|s| s.to_str()) == Some("safetensors") {
            tracing::info!("🔧 Loading DeepSeek Coder SafeTensors weights with F16 precision...");
            unsafe {
                Ok(VarBuilder::from_mmaped_safetensors(
                    weight_files,
                    DType::F16, // F16 for memory efficiency
                    device,
                )?)
            }
        } else {
            tracing::info!("🔧 Loading DeepSeek Coder PyTorch weights...");
            if weight_files.len() == 1 {
                let tensors_vec = candle_core::pickle::read_all(&weight_files[0])?;
                let tensors: std::collections::HashMap<String, candle_core::Tensor> =
                    tensors_vec.into_iter().collect();
                Ok(VarBuilder::from_tensors(tensors, DType::F16, device))
            } else {
                let mut all_tensors = std::collections::HashMap::new();
                for weight_file in weight_files {
                    let tensors_vec = candle_core::pickle::read_all(weight_file)?;
                    let tensors: std::collections::HashMap<String, candle_core::Tensor> =
                        tensors_vec.into_iter().collect();
                    all_tensors.extend(tensors);
                }
                Ok(VarBuilder::from_tensors(all_tensors, DType::F16, device))
            }
        }
    }

    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let total_start = std::time::Instant::now();
        
        // CODE GENERATION TEMPLATE: Optimized for DeepSeek Coder
        // DeepSeek Coder works best with clear instruction formatting
        let formatted_prompt = if prompt.contains("```") || prompt.to_lowercase().contains("function") || prompt.to_lowercase().contains("fn ") {
            // Already contains code or function keywords
            prompt.to_string()
        } else {
            // Add instruction formatting for better code generation
            format!("Please write code for the following request:\n\n{}\n\n```", prompt)
        };
        
        tracing::debug!("🎯 DeepSeek Coder prompt: '{}'", &formatted_prompt[..std::cmp::min(100, formatted_prompt.len())]);
        
        let tokenize_start = std::time::Instant::now();
        let input_ids = self
            .tokenizer
            .encode(formatted_prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("DeepSeek Coder tokenization failed: {}", e))?
            .get_ids()
            .to_vec();
        let tokenize_time = tokenize_start.elapsed();

        let input_tensor = Tensor::from_vec(
            input_ids.clone(),
            (1, input_ids.len()),
            &self.device,
        )?;

        // KV Cache for DeepSeek Coder with 16K context support
        let mut cache = Cache::new(true, DType::F16, &self.config, &self.device)?;
        let mut generated_tokens = Vec::with_capacity(max_tokens);
        let generation_start = std::time::Instant::now();
        
        // First forward pass
        let output = self.model.forward(&input_tensor, 0, &mut cache)?;
        let logits = output.squeeze(0)?;
        
        let next_token = logits.argmax(candle_core::D::Minus1)?;
        let mut token_id = next_token.to_scalar::<u32>()?;
        
        if !self.is_eos_token(token_id) {
            generated_tokens.push(token_id);
        }
        
        // Autoregressive generation loop
        for step in 1..max_tokens {
            if self.is_eos_token(token_id) {
                tracing::debug!("🏁 DeepSeek Coder hit EOS token at step {}", step);
                break;
            }
            
            let current_input = Tensor::new(&[token_id], &self.device)?.reshape((1, 1))?;
            let output = self.model.forward(&current_input, step, &mut cache)?;
            let logits = output.squeeze(0)?;
            
            let next_token = logits.argmax(candle_core::D::Minus1)?;
            token_id = next_token.to_scalar::<u32>()?;
            
            if !self.is_eos_token(token_id) {
                generated_tokens.push(token_id);
            }
        }
        
        let generation_time = generation_start.elapsed();
        
        let decode_start = std::time::Instant::now();
        let generated_text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("DeepSeek Coder decoding failed: {}", e))?;
        let decode_time = decode_start.elapsed();
        
        let total_time = total_start.elapsed();
        let tokens_per_second = generated_tokens.len() as f64 / generation_time.as_secs_f64();
        
        tracing::info!(
            "🚀 DeepSeek Coder Performance: {} tokens in {:?} ({:.1} tok/s) | Tokenize: {:?} | Generate: {:?} | Decode: {:?}",
            generated_tokens.len(),
            total_time,
            tokens_per_second,
            tokenize_time,
            generation_time,
            decode_time
        );
        
        Ok(generated_text)
    }

    fn is_eos_token(&self, token: u32) -> bool {
        if let Some(candle_transformers::models::llama::LlamaEosToks::Single(eos_id)) =
            &self.config.eos_token_id
        {
            token == *eos_id
        } else {
            token == 2 // Fallback EOS token
        }
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "DeepSeek-Coder-1.3B-Instruct".to_string(),
            version: "main".to_string(),
            parameters: 1_300_000_000, // 1.3B parameters
            memory_mb: self.estimate_memory_usage(),
            device: format!("{:?}", self.device),
            vocab_size: self.config.vocab_size,
            context_length: self.config.max_position_embeddings,
            model_type: "deepseek-coder".to_string(),
            architecture: "transformer".to_string(),
            precision: "f16".to_string(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        let params = 1_300_000_000; // 1.3B parameters
        let param_memory = params * 2; // F16 precision (2 bytes per param)
        let activation_memory = 128 * 1024 * 1024; // 128MB for activations and KV cache
        (param_memory + activation_memory) / (1024 * 1024)
    }
}

// ModelTrait implementation for DeepSeek Coder
#[async_trait]
impl ModelTrait for DeepSeekCoderModel {
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
        self.model_info()
    }

    fn device(&self) -> &candle_core::Device {
        &self.device
    }

    fn model_name(&self) -> &str {
        "deepseek-coder-1.3b-instruct"
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "generation" => true,
            "health_check" => true,
            "code_generation" => true,
            "instruction_following" => true,
            "multi_language_code" => true,
            "large_context" => true,
            "efficient_inference" => true,
            _ => false,
        }
    }
}

// Register DeepSeek Coder model with the registry
pub fn register_deepseek_coder_model() -> anyhow::Result<()> {
    use crate::models::registry::{global_registry, ModelRegistration};
    use std::sync::Arc;

    async fn deepseek_coder_factory() -> anyhow::Result<BoxedModel> {
        let model = DeepSeekCoderModel::load().await?;
        Ok(Box::new(model))
    }

    let registration = ModelRegistration {
        name: "deepseek-coder-1.3b-instruct".to_string(),
        aliases: vec![
            "deepseek-coder".to_string(),
            "deepseek".to_string(),
            "coder".to_string(),
            "deepseek-1.3b".to_string(),
            "ds-coder".to_string(),
        ],
        description: "DeepSeek Coder 1.3B Instruct - specialized for code generation, debugging, and explanation".to_string(),
        model_type: "deepseek-coder".to_string(),
        supported_features: vec![
            "generation".to_string(),
            "health_check".to_string(),
            "code_generation".to_string(),
            "instruction_following".to_string(),
            "multi_language_code".to_string(),
            "large_context".to_string(),
            "efficient_inference".to_string(),
        ],
        memory_requirements_mb: 3200, // ~3.2GB for 1.3B model
        factory: Arc::new(Box::new(|| Box::new(Box::pin(deepseek_coder_factory())))),
    };

    global_registry().register_model(registration)?;
    Ok(())
}