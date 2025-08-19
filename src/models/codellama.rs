// ARCHITECTURE: CodeLlama-7B Model Implementation - Code Generation Specialized Engine
//
// DESIGN PHILOSOPHY:
// CodeLlama-7B is Meta's code-specialized variant of Llama 2, optimized for:
// 1. CODE GENERATION: Superior performance on programming tasks vs general models
// 2. INSTRUCTION FOLLOWING: Fine-tuned for code completion and explanation
// 3. MULTI-LANGUAGE: Supports Python, JavaScript, C++, Java, PHP, TypeScript, C#, Bash
// 4. LARGER CONTEXT: Better handling of complex code structures and dependencies
//
// PERFORMANCE CHARACTERISTICS:
// - Model Size: 7B parameters (~13GB in F16 format)
// - Memory Usage: ~14GB total (weights + activations)
// - Inference Speed: 3-6 tokens/second on Apple Silicon (slower than TinyLlama due to size)
// - Context Length: 4096 tokens (2x TinyLlama)
// - Specialization: Code completion, debugging, explanation

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama};
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde_json::Value;
use std::fs;
use tokenizers::Tokenizer;
use std::sync::Arc;
use crate::models::traits::{ModelTrait, ModelInfo, BoxedModel, ModelHealthCheck};
use async_trait::async_trait;

/// CodeLlama-7B Model - Specialized Code Generation Engine
#[derive(Debug)]
pub struct CodeLlamaModel {
    model: Arc<Llama>,           // Thread-safe model reference
    tokenizer: Arc<Tokenizer>,   // Code-aware tokenizer
    pub device: Device,          // Compute device (preferably GPU for 7B model)
    config: Config,              // Model architecture configuration
}

impl CodeLlamaModel {
    pub async fn load() -> Result<Self> {
        tracing::info!("🚀 Starting CodeLlama-7B model loading...");

        // GPU ACCELERATION: Critical for 7B models
        // 7B models are significantly slower on CPU - GPU highly recommended
        let device = if candle_core::utils::metal_is_available() {
            tracing::info!("🚀 Metal GPU available, using acceleration (recommended for 7B models)");
            match Device::new_metal(0) {
                Ok(d) => {
                    tracing::info!("✅ Successfully initialized Metal device for CodeLlama-7B");
                    d
                }
                Err(e) => {
                    tracing::warn!("❌ Failed to initialize Metal: {}, falling back to CPU (performance will be significantly slower)", e);
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
            tracing::warn!("🖥️  Using CPU for CodeLlama-7B - expect slower performance (~1-2 tok/s)");
            Device::Cpu
        };
        
        tracing::info!("📱 CodeLlama device: {:?}", device);

        // Load CodeLlama components
        let (model, tokenizer, config) = Self::load_model_components(&device).await?;

        tracing::info!("✅ CodeLlama-7B model loaded successfully");
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
        let api = if let Ok(token) = std::env::var("HF_TOKEN") {
            tracing::info!("🔑 Using HuggingFace token for CodeLlama authentication");
            hf_hub::api::sync::ApiBuilder::new()
                .with_token(Some(token))
                .build()?
        } else {
            tracing::warn!("⚠️ No HF_TOKEN found for CodeLlama, using anonymous access");
            Api::new()?
        };
        let repo = api.repo(Repo::with_revision(
            "codellama/CodeLlama-7b-hf".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        tracing::info!("📥 Downloading CodeLlama configuration files...");
        let config_file = repo
            .get("config.json")
            .map_err(|e| anyhow::anyhow!("Failed to download CodeLlama config.json: {}", e))?;

        let tokenizer_file = repo
            .get("tokenizer.json")
            .map_err(|e| anyhow::anyhow!("Failed to download CodeLlama tokenizer.json: {}", e))?;

        let config_json = fs::read_to_string(&config_file)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load CodeLlama tokenizer: {}", e))?;

        let config = Self::parse_codellama_config(&config_json)?;
        tracing::info!(
            "📊 CodeLlama Config: vocab_size={}, hidden_size={}, num_layers={}, context_length={}",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.max_position_embeddings
        );

        tracing::info!("📥 Downloading CodeLlama-7B weights (~13GB)...");
        let weight_files = Self::download_codellama_weights(&repo)?;
        let vars = Self::load_weights(&weight_files, &device)?;

        tracing::info!("🏗️  Building CodeLlama-7B model graph...");
        let model = Llama::load(vars, &config)?;

        tracing::info!("✅ CodeLlama-7B model components loaded successfully");
        Ok((model, tokenizer, config))
    }

    fn parse_codellama_config(config_json: &str) -> Result<Config> {
        let config: Value = serde_json::from_str(config_json)?;

        // CodeLlama-7B specific parameters
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(32016) as usize; // CodeLlama has slightly larger vocab
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(4096) as usize;
        let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(11008) as usize;
        let num_hidden_layers = config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let num_attention_heads = config["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let num_key_value_heads = config
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(32); // CodeLlama uses full attention heads
        let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-5);
        let rope_theta = config
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let max_position_embeddings = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize; // 4K context length

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

    fn download_codellama_weights(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<std::path::PathBuf>> {
        // CodeLlama-7B weight file patterns
        let possible_patterns = vec![
            // SafeTensors (preferred)
            (1..=3)
                .map(|i| format!("model-{i:05}-of-00003.safetensors"))
                .collect::<Vec<_>>(),
            vec!["model.safetensors".to_string()],
            // PyTorch fallback
            (1..=3)
                .map(|i| format!("pytorch_model-{i:05}-of-00003.bin"))
                .collect::<Vec<_>>(),
            vec!["pytorch_model.bin".to_string()],
        ];

        for pattern in possible_patterns {
            let mut pattern_files = Vec::new();
            let mut all_found = true;

            for filename in &pattern {
                match repo.get(filename) {
                    Ok(path) => {
                        tracing::info!("✅ Downloaded CodeLlama weight file: {}", filename);
                        pattern_files.push(path);
                    }
                    Err(_) => {
                        all_found = false;
                        break;
                    }
                }
            }

            if all_found && !pattern_files.is_empty() {
                tracing::info!("✅ Successfully downloaded {} CodeLlama weight file(s)", pattern_files.len());
                return Ok(pattern_files);
            }
        }

        Err(anyhow::anyhow!("No CodeLlama weight files found"))
    }

    fn load_weights<'a>(
        weight_files: &'a [std::path::PathBuf],
        device: &'a Device,
    ) -> Result<VarBuilder<'a>> {
        if weight_files[0].extension().and_then(|s| s.to_str()) == Some("safetensors") {
            tracing::info!("🔧 Loading CodeLlama SafeTensors weights with F16 precision...");
            unsafe {
                Ok(VarBuilder::from_mmaped_safetensors(
                    weight_files,
                    DType::F16, // F16 for memory efficiency with 7B model
                    device,
                )?)
            }
        } else {
            tracing::info!("🔧 Loading CodeLlama PyTorch weights...");
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
        
        // CODE GENERATION TEMPLATE: Optimized for CodeLlama
        // CodeLlama performs better with clear instruction formatting
        let formatted_prompt = if prompt.contains("```") || prompt.to_lowercase().contains("code") {
            // Already formatted or contains code
            prompt.to_string()
        } else {
            // Add instruction formatting for better code generation
            format!("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:", prompt)
        };
        
        tracing::debug!("🎯 CodeLlama prompt: '{}'", &formatted_prompt[..std::cmp::min(100, formatted_prompt.len())]);
        
        let tokenize_start = std::time::Instant::now();
        let input_ids = self
            .tokenizer
            .encode(formatted_prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("CodeLlama tokenization failed: {}", e))?
            .get_ids()
            .to_vec();
        let tokenize_time = tokenize_start.elapsed();

        let input_tensor = Tensor::from_vec(
            input_ids.clone(),
            (1, input_ids.len()),
            &self.device,
        )?;

        // KV Cache for CodeLlama with 4K context support
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
                tracing::debug!("🏁 CodeLlama hit EOS token at step {}", step);
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
            .map_err(|e| anyhow::anyhow!("CodeLlama decoding failed: {}", e))?;
        let decode_time = decode_start.elapsed();
        
        let total_time = total_start.elapsed();
        let tokens_per_second = generated_tokens.len() as f64 / generation_time.as_secs_f64();
        
        tracing::info!(
            "🚀 CodeLlama Performance: {} tokens in {:?} ({:.1} tok/s) | Tokenize: {:?} | Generate: {:?} | Decode: {:?}",
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
            name: "CodeLlama-7B-HF".to_string(),
            version: "main".to_string(),
            parameters: 6_700_000_000, // 6.7B parameters
            memory_mb: self.estimate_memory_usage(),
            device: format!("{:?}", self.device),
            vocab_size: self.config.vocab_size,
            context_length: self.config.max_position_embeddings,
            model_type: "codellama".to_string(),
            architecture: "transformer".to_string(),
            precision: "f16".to_string(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        let params = 6_700_000_000; // 6.7B parameters
        let param_memory = params * 2; // F16 precision (2 bytes per param)
        let activation_memory = 64 * 1024 * 1024; // 64MB for activations and KV cache
        (param_memory + activation_memory) / (1024 * 1024)
    }
}

// ModelTrait implementation for CodeLlama
#[async_trait]
impl ModelTrait for CodeLlamaModel {
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
        "codellama-7b-instruct"
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "generation" => true,
            "health_check" => true,
            "code_generation" => true,
            "instruction_following" => true,
            "multi_language" => true,
            "large_context" => true,
            _ => false,
        }
    }
}

// Register CodeLlama model with the registry
pub fn register_codellama_model() -> anyhow::Result<()> {
    use crate::models::registry::{global_registry, ModelRegistration};
    use std::sync::Arc;

    async fn codellama_factory() -> anyhow::Result<BoxedModel> {
        let model = CodeLlamaModel::load().await?;
        Ok(Box::new(model))
    }

    let registration = ModelRegistration {
        name: "codellama-7b-instruct".to_string(),
        aliases: vec![
            "codellama".to_string(),
            "codellama-7b".to_string(),
            "code-llama".to_string(),
            "code".to_string(),
            "coding".to_string(),
        ],
        description: "Meta's CodeLlama 7B HF - specialized for code generation and programming tasks".to_string(),
        model_type: "codellama".to_string(),
        supported_features: vec![
            "generation".to_string(),
            "health_check".to_string(),
            "code_generation".to_string(),
            "instruction_following".to_string(),
            "multi_language".to_string(),
            "large_context".to_string(),
        ],
        memory_requirements_mb: 14000, // ~14GB for 7B model
        factory: Arc::new(Box::new(|| Box::new(Box::pin(codellama_factory())))),
    };

    global_registry().register_model(registration)?;
    Ok(())
}