// ARCHITECTURE: Generic Llama Model Implementation - Flexible Multi-Model Support
//
// DESIGN PHILOSOPHY:
// This module implements a generic Llama model loader that can work with any
// Llama-based model from HuggingFace, providing:
// 1. FLEXIBLE MODEL LOADING: Support for any Llama-based model (Llama-2, Llama-3.2, etc.)
// 2. AUTOMATIC DISCOVERY: Auto-discovery of safetensors weight files
// 3. DEVICE OPTIMIZATION: Automatic GPU/CPU device selection with fallbacks
// 4. INCREMENTAL GENERATION: Efficient token-by-token generation with KV caching
//
// PERFORMANCE CHARACTERISTICS:
// - Model Size: Variable (1B-70B+ parameters depending on chosen model)
// - Memory Usage: Auto-scaling based on model size and precision
// - Context Length: Variable (2K-32K+ tokens depending on model)
// - Precision: F16 for GPU, F32 for CPU

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use serde_json::Value;
use candle_transformers::models::llama as llama_mod;
use llama_mod::{Config as LlamaConfig, Llama, Cache as LlamaCache};
use hf_hub::{api::tokio::{Api, ApiBuilder}, Repo, RepoType};
use std::{env, sync::Arc};
use tokio::sync::Mutex;
use tokenizers::Tokenizer;
use crate::models::traits::{ModelTrait, ModelInfo, BoxedModel};
use async_trait::async_trait;

/// Generic Llama Model - Flexible Multi-Model Support
#[derive(Debug)]
pub struct GenericLlamaModel {
    model: Arc<Mutex<LlamaModelWrapper>>,
    tokenizer: Arc<Tokenizer>,
    pub device: Device,
    config: LlamaConfig,
    repo_name: String,
}

// Wrapper for Llama model with its cache
#[derive(Debug)]
struct LlamaModelWrapper {
    model: Llama,
    cache: LlamaCache,
    config: LlamaConfig,
    dtype: DType,
}

impl LlamaModelWrapper {
    fn forward(&mut self, input: &Tensor, seq_offset: usize) -> Result<Tensor> {
        self.model.forward(input, seq_offset, &mut self.cache)
            .map_err(|e| anyhow::anyhow!("Llama forward error: {}", e))
    }
    
    fn reset_cache(&mut self, device: &Device) -> Result<()> {
        self.cache = LlamaCache::new(true, self.dtype, &self.config, device)?;
        Ok(())
    }
}

impl GenericLlamaModel {
    pub async fn load_from_repo(repo_name: &str) -> Result<Self> {
        tracing::info!("🚀 Starting Generic Llama model loading from: {}", repo_name);

        // DEVICE SELECTION: Prioritize GPU for better performance
        let device = Self::create_device()?;
        tracing::info!("📱 Selected device: {:?}", device);

        // Load model components
        let (model, tokenizer, config) = Self::load_model(&device, repo_name).await?;

        tracing::info!("✅ Generic Llama model loaded successfully: {}", repo_name);
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            config,
            repo_name: repo_name.to_string(),
        })
    }

    pub async fn load() -> Result<Self> {
        // Default to Llama-3.2-1B for compatibility
        Self::load_from_repo("meta-llama/Llama-3.2-1B").await
    }

    fn create_device() -> Result<Device> {
        // Prioritize Metal > CUDA > CPU
        if candle_core::utils::metal_is_available() {
            tracing::info!("🚀 Metal GPU available, using acceleration");
            match Device::new_metal(0) {
                Ok(d) => {
                    tracing::info!("✅ Successfully initialized Metal device");
                    Ok(d)
                }
                Err(e) => {
                    tracing::warn!("❌ Failed to initialize Metal: {}, trying CUDA", e);
                    Self::try_cuda_or_cpu()
                }
            }
        } else {
            Self::try_cuda_or_cpu()
        }
    }

    fn try_cuda_or_cpu() -> Result<Device> {
        if candle_core::utils::cuda_is_available() {
            tracing::info!("🚀 CUDA GPU available, using acceleration");
            match Device::new_cuda(0) {
                Ok(cuda_device) => {
                    tracing::info!("✅ Using device: CUDA (device 0)");
                    Ok(cuda_device)
                }
                Err(e) => {
                    tracing::warn!("❌ Failed to initialize CUDA: {}, falling back to CPU", e);
                    Ok(Device::Cpu)
                }
            }
        } else {
            tracing::info!("🖥️  Using CPU for inference");
            Ok(Device::Cpu)
        }
    }

    async fn load_model(device: &Device, repo_str: &str) -> Result<(LlamaModelWrapper, Tokenizer, LlamaConfig)> {
        // Check for HF_TOKEN environment variable
        let _token = env::var("HF_TOKEN").map_err(|_| {
            anyhow::anyhow!(
                "Please set your Hugging Face token as an environment variable:\n\
                 export HF_TOKEN=your_token_here\n\n\
                 You can get a token from: https://huggingface.co/settings/tokens\n\
                 Make sure to accept the model license on Hugging Face"
            )
        })?;

        tracing::info!("🔑 Using HuggingFace authentication for: {}", repo_str);

        let api = if let Ok(token) = std::env::var("HF_TOKEN") {
            ApiBuilder::new().with_token(Some(token)).build()?
        } else {
            Api::new()?
        };

        let repo = api.repo(Repo::with_revision(
            repo_str.to_string(), 
            RepoType::Model, 
            "main".to_string()
        ));

        // Load tokenizer
        tracing::info!("📥 Downloading tokenizer for {}...", repo_str);
        let tokenizer_filename = repo.get("tokenizer.json").await?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        tracing::info!("✅ Tokenizer loaded successfully");

        // Load config
        tracing::info!("📥 Downloading model config...");
        let config_file = repo.get("config.json").await
            .map_err(|e| anyhow::anyhow!("Failed to download config.json: {}", e))?;
        let config_content = std::fs::read_to_string(config_file)?;
        let config = Self::parse_llama_config(&config_content)?;
        
        tracing::info!(
            "🔎 Parsed Llama config: vocab_size={}, hidden_size={}, num_layers={}", 
            config.vocab_size, config.hidden_size, config.num_hidden_layers
        );

        // Auto-discover safetensors files
        tracing::info!("📥 Discovering model weight files for {}...", repo_str);
        let safetensors_files = Self::discover_safetensors(&repo).await?;
        
        // Print file info for debugging
        for f in &safetensors_files {
            if let Ok(m) = std::fs::metadata(f) {
                tracing::info!("📦 {:?} ({} bytes)", f, m.len());
            }
        }

        // Choose dtype based on device
        let chosen_dtype = if matches!(device, Device::Cpu) { DType::F32 } else { DType::F16 };
        tracing::info!("⚙️  Using tensor DType: {:?} for device: {:?}", chosen_dtype, device);

        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&safetensors_files, chosen_dtype, device)? 
        };
        tracing::info!("✅ VarBuilder created from safetensors");

        let model = Llama::load(vb, &config)?;
        let cache = LlamaCache::new(true, chosen_dtype, &config, device)?;
        tracing::info!("✅ Llama model built successfully");

        Ok((
            LlamaModelWrapper { model, cache, config: config.clone(), dtype: chosen_dtype }, 
            tokenizer, 
            config
        ))
    }

    async fn discover_safetensors(repo: &hf_hub::api::tokio::ApiRepo) -> Result<Vec<std::path::PathBuf>> {
        let mut safetensors_files = Vec::new();
        
        // Try common patterns
        let candidates = vec![
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors", 
            "model.safetensors",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ];
        
        for candidate in candidates {
            if let Ok(path) = repo.get(candidate).await {
                tracing::info!("✅ Found weight file: {}", candidate);
                safetensors_files.push(path);
            }
        }

        if safetensors_files.is_empty() {
            return Err(anyhow::anyhow!("No safetensors weight files found"));
        }

        Ok(safetensors_files)
    }

    // Helper function to create Llama Config from HuggingFace config.json
    fn parse_llama_config(config_json: &str) -> Result<LlamaConfig> {
        let config: Value = serde_json::from_str(config_json)?;
        
        // Extract values from the HuggingFace config
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(2048) as usize;
        let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(5632) as usize;
        let num_hidden_layers = config["num_hidden_layers"].as_u64().unwrap_or(22) as usize;
        let num_attention_heads = config["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let num_key_value_heads = config.get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);
        let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-5);
        let rope_theta = config.get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let max_position_embeddings = config.get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

        Ok(LlamaConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps,
            rope_theta: rope_theta as f32,
            max_position_embeddings,
            bos_token_id: Some(config.get("bos_token_id").and_then(|v| v.as_i64()).unwrap_or(1) as u32),
            eos_token_id: Some(llama_mod::LlamaEosToks::Single(
                config.get("eos_token_id").and_then(|v| v.as_i64()).unwrap_or(2) as u32
            )),
            rope_scaling: None,
            tie_word_embeddings: config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false),
            use_flash_attn: false,
        })
    }

    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let total_start = std::time::Instant::now();
        
        // Tokenize input
        let tokenize_start = std::time::Instant::now();
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        let token_ids = tokens.get_ids().iter().map(|&id| id as u32).collect::<Vec<_>>();
        let tokenize_time = tokenize_start.elapsed();
        
        tracing::debug!("✅ Input tokenized: {} tokens", token_ids.len());

        // Generate text using incremental approach
        let generation_start = std::time::Instant::now();
        let mut model = self.model.lock().await;
        model.reset_cache(&self.device)?;
        
        let generated_text = self.greedy_decode_incremental(&mut *model, &token_ids, max_tokens)?;
        drop(model); // Release lock early
        
        let generation_time = generation_start.elapsed();
        let total_time = total_start.elapsed();
        
        // Calculate performance metrics
        let tokens_generated = generated_text.split_whitespace().count();
        let tokens_per_second = tokens_generated as f64 / generation_time.as_secs_f64();
        
        tracing::info!(
            "🚀 Generic Llama Performance: {} tokens in {:?} ({:.1} tok/s) | Tokenize: {:?} | Generate: {:?}",
            tokens_generated,
            total_time,
            tokens_per_second,
            tokenize_time,
            generation_time
        );
        
        Ok(generated_text)
    }

    fn greedy_decode_incremental(
        &self,
        lm: &mut LlamaModelWrapper,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<String> {
        // First pass with the whole prompt at offset 0
        let mut ctx_len = input_ids.len();
        let mut generated = Vec::<u32>::new();

        let input = Tensor::from_vec(input_ids.to_vec(), (1, ctx_len), &self.device)?;
        // Warm up cache with prompt
        let _ = lm.forward(&input, 0)?;

        for _step in 0..max_new_tokens {
            // Feed only the previous generated token (or last prompt token on first step)
            let last_tok = if generated.is_empty() {
                *input_ids.last().unwrap()
            } else {
                *generated.last().unwrap()
            };
            let t = Tensor::from_vec(vec![last_tok], (1, 1), &self.device)?;

            // Forward with current context length as offset
            let logits = lm.forward(&t, ctx_len)?;
            ctx_len += 1;

            // Handle different logits shapes
            let logits_dims = logits.dims();
            let last_logits = if logits_dims.len() == 2 {
                // [batch, vocab] format
                logits.i((0,))?
            } else if logits_dims.len() == 3 {
                // [batch, seq, vocab] format
                logits.i((0, 0))?
            } else {
                return Err(anyhow::anyhow!("Unexpected logits shape: {:?}", logits_dims));
            };
            
            // Apply temperature and top-p sampling to avoid repetition
            let next_id = self.sample_with_temperature(&last_logits, 0.8, 0.7)?;

            if self.is_eos_token(next_id) { 
                tracing::debug!("🏁 Hit EOS token, stopping generation");
                break; 
            }

            // Check for repetition patterns to improve quality
            if self.detect_repetition(&generated, next_id)? {
                tracing::debug!("🔄 Repetition detected, stopping generation");
                break;
            }

            generated.push(next_id);
        }

        let text = self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
        Ok(text)
    }

    fn is_eos_token(&self, token_id: u32) -> bool {
        // Check against config's EOS token
        if let Some(llama_mod::LlamaEosToks::Single(eos_id)) = &self.config.eos_token_id {
            return token_id == *eos_id;
        }
        
        // Common EOS tokens for various Llama models
        if let Some(eos_id) = self.tokenizer.token_to_id("<end_of_turn>") {
            if token_id == eos_id { return true; }
        }
        if let Some(eos_id) = self.tokenizer.token_to_id("</s>") {
            if token_id == eos_id { return true; }
        }
        if let Some(eos_id) = self.tokenizer.token_to_id("<|endoftext|>") {
            if token_id == eos_id { return true; }
        }
        
        // Common EOS token IDs as fallback
        matches!(token_id, 1 | 2 | 106 | 107)
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: format!("Generic-Llama ({})", self.repo_name),
            version: "main".to_string(),
            parameters: self.estimate_parameters(),
            memory_mb: self.estimate_memory_usage(),
            device: format!("{:?}", self.device),
            vocab_size: self.config.vocab_size,
            context_length: self.config.max_position_embeddings,
            model_type: "llama".to_string(),
            architecture: "transformer".to_string(),
            precision: if matches!(self.device, Device::Cpu) { "f32" } else { "f16" }.to_string(),
        }
    }

    fn sample_with_temperature(&self, logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32> {
        use candle_nn::ops::softmax;
        use rand::prelude::*;
        use rand::rng;
        
        // Apply temperature
        let scaled_logits = (logits / temperature)?;
        
        // Apply softmax to get probabilities
        let probs = softmax(&scaled_logits, candle_core::D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;
        
        // Apply top-p (nucleus) sampling
        let mut indexed_probs: Vec<(usize, f32)> = probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut cumsum = 0.0;
        let mut cutoff = indexed_probs.len();
        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }
        
        // Sample from the truncated distribution
        let mut rng = rng();
        let total_mass: f32 = indexed_probs[..cutoff].iter().map(|(_, p)| p).sum();
        let mut rand_val = rng.random::<f32>() * total_mass;
        
        for &(idx, prob) in &indexed_probs[..cutoff] {
            rand_val -= prob;
            if rand_val <= 0.0 {
                return Ok(idx as u32);
            }
        }
        
        // Fallback to the most likely token
        Ok(indexed_probs[0].0 as u32)
    }

    fn detect_repetition(&self, generated: &[u32], next_id: u32) -> Result<bool> {
        if generated.len() < 6 { return Ok(false); }
        
        // Check for immediate repetition (same token repeated)
        if generated.len() >= 3 {
            let last_3 = &generated[generated.len()-3..];
            if last_3.iter().all(|&token| token == next_id) {
                return Ok(true);
            }
        }
        
        // Check for pattern repetition (last 4 tokens repeated)
        if generated.len() >= 8 {
            let last_4 = &generated[generated.len()-4..];
            let prev_4 = &generated[generated.len()-8..generated.len()-4];
            if last_4 == prev_4 {
                return Ok(true);
            }
        }
        
        // Check for phrase repetition using decoded text
        if generated.len() >= 12 {
            let recent_text = self.tokenizer
                .decode(&generated[generated.len()-12..], true)
                .unwrap_or_default();
            
            // Simple heuristic: if we see the same phrase pattern
            if recent_text.matches("Mbps").count() > 2 ||
               recent_text.matches("policy").count() > 2 ||
               recent_text.matches("connection").count() > 2 {
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    fn estimate_parameters(&self) -> usize {
        // Estimate based on config - rough calculation
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_hidden_layers;
        let vocab_size = self.config.vocab_size;
        let intermediate_size = self.config.intermediate_size;
        
        // Rough parameter estimation for Llama architecture
        let embedding_params = vocab_size * hidden_size;
        let attention_params = num_layers * hidden_size * hidden_size * 4; // Q, K, V, O projections
        let mlp_params = num_layers * (hidden_size * intermediate_size * 2); // up and down projections
        
        embedding_params + attention_params + mlp_params
    }

    fn estimate_memory_usage(&self) -> usize {
        let params = self.estimate_parameters();
        let bytes_per_param = if matches!(self.device, Device::Cpu) { 4 } else { 2 }; // F32 vs F16
        let param_memory = params * bytes_per_param;
        let activation_memory = 64 * 1024 * 1024; // 64MB estimate
        (param_memory + activation_memory) / (1024 * 1024)
    }
}

// ModelTrait implementation for Generic Llama
#[async_trait]
impl ModelTrait for GenericLlamaModel {
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
        "llama-generic"
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "generation" => true,
            "health_check" => true,
            "incremental_generation" => true,
            "flexible_repo" => true,
            "auth_required" => true,
            _ => false,
        }
    }
}

// Register Generic Llama model
pub fn register_generic_llama_model() -> anyhow::Result<()> {
    use crate::models::registry::{global_registry, ModelRegistration};
    use std::sync::Arc;

    async fn generic_llama_factory() -> anyhow::Result<BoxedModel> {
        let model = GenericLlamaModel::load().await?;
        Ok(Box::new(model))
    }

    let registration = ModelRegistration {
        name: "llama-generic".to_string(),
        aliases: vec![
            "llama3".to_string(),
            "llama-3.2".to_string(),
            "generic-llama".to_string(),
            "flexible-llama".to_string(),
        ],
        description: "Generic Llama model loader - supports any HuggingFace Llama model with authentication".to_string(),
        model_type: "llama".to_string(),
        supported_features: vec![
            "generation".to_string(),
            "health_check".to_string(),
            "incremental_generation".to_string(),
            "flexible_repo".to_string(),
            "auth_required".to_string(),
        ],
        memory_requirements_mb: 4000, // Variable based on model size
        factory: Arc::new(Box::new(|| Box::new(Box::pin(generic_llama_factory())))),
    };

    global_registry().register_model(registration)?;
    Ok(())
}