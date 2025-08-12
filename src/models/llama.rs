use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama};
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde_json::Value;
use std::fs;
use tokenizers::Tokenizer;
use std::sync::Arc;

pub struct TinyLlamaModel {
    model: Arc<Llama>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    config: Config,
}

impl TinyLlamaModel {
    pub async fn load() -> Result<Self> {
        tracing::info!("Starting TinyLlama model loading...");

        // Prioritize GPU with fallback to CPU
        let device = if candle_core::utils::metal_is_available() {
            tracing::info!("🚀 Metal available, using GPU acceleration");
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
        } else {
            tracing::info!("❌ Metal not available, using CPU");
            Device::Cpu
        };
        
        tracing::info!("📱 Final selected device: {:?}", device);

        // YOUR EXISTING MODEL DOWNLOAD CODE
        let (model, tokenizer, config) = Self::load_model_components(&device).await?;

        tracing::info!("TinyLlama model loaded successfully");
        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            device,
            config,
        })
    }

    async fn load_model_components(device: &Device) -> Result<(Llama, Tokenizer, Config)> {
        // YOUR EXISTING DOWNLOAD LOGIC (with async wrapper)
        let device_clone = device.clone();
        let (model, tokenizer, config) =
            tokio::task::spawn_blocking(move || Self::load_model_sync(device_clone)).await??;

        Ok((model, tokenizer, config))
    }

    fn load_model_sync(device: Device) -> Result<(Llama, Tokenizer, Config)> {
        // PASTE YOUR EXISTING load_chat_model() LOGIC HERE
        // But return (model, tokenizer, config) tuple instead of ChatModel

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        tracing::info!("Downloading configuration files...");
        let config_file = repo
            .get("config.json")
            .map_err(|e| anyhow::anyhow!("Failed to download config.json: {}", e))?;

        let tokenizer_file = repo
            .get("tokenizer.json")
            .map_err(|e| anyhow::anyhow!("Failed to download tokenizer.json: {}", e))?;

        let config_json = fs::read_to_string(&config_file)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let config = Self::parse_llama_config(&config_json)?;
        tracing::info!(
            "Parsed config: vocab_size={}, hidden_size={}, num_layers={}",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers
        );

        // YOUR EXISTING WEIGHT LOADING LOGIC
        tracing::info!("Downloading model weights...");
        let weight_files = Self::download_weight_files(&repo)?;
        let vars = Self::load_weights(&weight_files, &device)?;

        tracing::info!("Building model graph...");
        let model = Llama::load(vars, &config)?;

        Ok((model, tokenizer, config))
    }

    fn parse_llama_config(config_json: &str) -> Result<Config> {
        // YOUR EXISTING CONFIG PARSING CODE
        let config: Value = serde_json::from_str(config_json)?;

        let vocab_size = config["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(2048) as usize;
        let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(5632) as usize;
        let num_hidden_layers = config["num_hidden_layers"].as_u64().unwrap_or(22) as usize;
        let num_attention_heads = config["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let num_key_value_heads = config
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);
        let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-5);
        let rope_theta = config
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let max_position_embeddings = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

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

    fn download_weight_files(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<std::path::PathBuf>> {
        // YOUR EXISTING WEIGHT FILE DISCOVERY LOGIC
        let possible_patterns = vec![
            vec!["model.safetensors".to_string()],
            (1..=2)
                .map(|i| format!("model-{i:05}-of-00002.safetensors"))
                .collect::<Vec<_>>(),
            vec!["pytorch_model.bin".to_string()],
            (1..=2)
                .map(|i| format!("pytorch_model-{i:05}-of-00002.bin"))
                .collect::<Vec<_>>(),
        ];

        for pattern in possible_patterns {
            let mut pattern_files = Vec::new();
            let mut all_found = true;

            for filename in &pattern {
                match repo.get(filename) {
                    Ok(path) => {
                        tracing::debug!("Found weight file: {}", filename);
                        pattern_files.push(path);
                    }
                    Err(_) => {
                        all_found = false;
                        break;
                    }
                }
            }

            if all_found && !pattern_files.is_empty() {
                tracing::info!("Successfully found {} weight file(s)", pattern_files.len());
                return Ok(pattern_files);
            }
        }

        Err(anyhow::anyhow!("No model weight files found"))
    }

    fn load_weights<'a>(
        weight_files: &'a [std::path::PathBuf],
        device: &'a Device,
    ) -> Result<VarBuilder<'a>> {
        // YOUR EXISTING WEIGHT LOADING LOGIC
        if weight_files[0].extension().and_then(|s| s.to_str()) == Some("safetensors") {
            tracing::info!("Loading safetensors weights...");
            unsafe {
                Ok(VarBuilder::from_mmaped_safetensors(
                    weight_files,
                    DType::F16,
                    device,
                )?)
            }
        } else {
            tracing::info!("Loading PyTorch weights...");
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
        
        // Format the input as a chat prompt
        let formatted_prompt = format!("User: {prompt}\nAssistant:");
        
        // Tokenize input (measure timing)
        let tokenize_start = std::time::Instant::now();
        let input_ids = self
            .tokenizer
            .encode(formatted_prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?
            .get_ids()
            .to_vec();
        let tokenize_time = tokenize_start.elapsed();

        // Convert to 2D tensor with batch dimension
        let input_tensor = Tensor::from_vec(
            input_ids.clone(),
            (1, input_ids.len()),
            &self.device,
        )?;

        // Create cache with F16 to match model weights
        let mut cache = Cache::new(true, DType::F16, &self.config, &self.device)?;
        
        // Generate tokens with timing
        let mut generated_tokens = Vec::with_capacity(max_tokens);
        let generation_start = std::time::Instant::now();
        
        // First forward pass with full prompt
        let output = self.model.forward(&input_tensor, 0, &mut cache)?;
        let logits = output.squeeze(0)?;
        let next_token = logits.argmax(candle_core::D::Minus1)?;
        let mut token_id = next_token.to_scalar::<u32>()?;
        
        if !self.is_eos_token(token_id) {
            generated_tokens.push(token_id);
        }
        
        // Subsequent forward passes with single tokens
        for step in 1..max_tokens {
            if self.is_eos_token(token_id) {
                break;
            }
            
            // Create single token tensor
            let current_input = Tensor::from_vec(
                vec![token_id],
                (1, 1),
                &self.device,
            )?;
            
            let output = self.model.forward(&current_input, step, &mut cache)?;
            let logits = output.squeeze(0)?;
            let next_token = logits.argmax(candle_core::D::Minus1)?;
            token_id = next_token.to_scalar::<u32>()?;
            
            if !self.is_eos_token(token_id) {
                generated_tokens.push(token_id);
            }
        }
        
        let generation_time = generation_start.elapsed();

        // Decode tokens (measure timing)
        let decode_start = std::time::Instant::now();
        let generated_text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        let decode_time = decode_start.elapsed();
        
        let total_time = total_start.elapsed();
        let tokens_per_second = generated_tokens.len() as f64 / generation_time.as_secs_f64();
        
        tracing::info!(
            "🚀 Performance: {} tokens in {:?} ({:.1} tok/s) | Tokenize: {:?} | Generate: {:?} | Decode: {:?}",
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
        // Check against the config's EOS token
        if let Some(candle_transformers::models::llama::LlamaEosToks::Single(eos_id)) =
            &self.config.eos_token_id
        {
            token == *eos_id
        } else {
            token == 2 // Fallback to common EOS token
        }
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "TinyLlama-1.1B-Chat".to_string(),
            version: "v1.0".to_string(),
            parameters: self.estimate_parameters(),
            memory_mb: self.estimate_memory_usage(),
            device: format!("{:?}", self.device),
            vocab_size: self.config.vocab_size,
            context_length: self.config.max_position_embeddings,
        }
    }

    fn estimate_parameters(&self) -> usize {
        // Corrected parameter count for TinyLlama-1.1B
        1_100_000_000 // Actual parameter count
    }

    fn estimate_memory_usage(&self) -> usize {
        let params = self.estimate_parameters();
        let param_memory = params * 2; // FP16
        let activation_memory = 2 * 1024 * 1024; // 2MB estimate for activations
        (param_memory + activation_memory) / (1024 * 1024)
    }
}

#[derive(serde::Serialize, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub parameters: usize,
    pub memory_mb: usize,
    pub device: String,
    pub vocab_size: usize,
    pub context_length: usize,
}
