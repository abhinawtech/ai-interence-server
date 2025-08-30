// ARCHITECTURE: Llama-2-7B-Chat Model Implementation - Production Chat Model
//
// DESIGN PHILOSOPHY:
// This module implements Meta's Llama-2-7B-Chat, a state-of-the-art conversational AI model:
// 1. OPTIMIZED CHAT PERFORMANCE: Fine-tuned specifically for conversational interactions
// 2. ROBUST SCALING: 7B parameters provide excellent quality/performance balance
// 3. PRODUCTION READY: Proven stability and reliability for enterprise deployment
// 4. AUTHENTIC CHAT FORMAT: Uses official Llama-2 chat template for optimal results
//
// PERFORMANCE CHARACTERISTICS:
// - Model Size: 7B parameters (~13GB in F16 format)
// - Memory Usage: ~14GB total (weights + activations + KV cache)
// - Context Length: 4096 tokens maximum
// - Precision: F16 for GPU, F32 for CPU
// - Chat Template: Official Meta Llama-2-Chat format with system/user/assistant roles

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama};
use hf_hub::{api::tokio::ApiBuilder, Repo, RepoType};
use serde_json::Value;
use std::{env, sync::Arc};
use tokio::sync::Mutex;
use tokenizers::Tokenizer;
use crate::models::traits::{ModelTrait, ModelInfo, BoxedModel};
use async_trait::async_trait;
use rand::{prelude::*};

/// Llama-2-7B-Chat Model - Production Conversational AI
#[derive(Debug)]
pub struct Llama2ChatModel {
    model: Arc<Mutex<LlamaWrapper>>,
    tokenizer: Arc<Tokenizer>,
    pub device: Device,
    config: Config,
}

// Wrapper to handle model and cache together
#[derive(Debug)]
struct LlamaWrapper {
    model: Llama,
    cache: Cache,
    config: Config,
    dtype: DType,
}

impl LlamaWrapper {
    fn forward(&mut self, input: &Tensor, seq_offset: usize) -> Result<Tensor> {
        self.model.forward(input, seq_offset, &mut self.cache)
            .map_err(|e| anyhow::anyhow!("Llama-2 forward error: {}", e))
    }
    
    fn reset_cache(&mut self, device: &Device) -> Result<()> {
        self.cache = Cache::new(true, self.dtype, &self.config, device)?;
        Ok(())
    }
}

impl Llama2ChatModel {
    pub async fn load() -> Result<Self> {
        tracing::info!("🚀 Starting Llama-2-7B-Chat model loading...");
        
        let device = Self::create_device()?;
        tracing::info!("📱 Selected device: {:?}", device);

        let (model, tokenizer, config) = Self::load_model_components(&device).await?;

        tracing::info!("✅ Llama-2-7B-Chat model loaded successfully");
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            config,
        })
    }

    fn create_device() -> Result<Device> {
        // Prioritize Metal > CUDA > CPU for Llama-2-7B
        if candle_core::utils::metal_is_available() {
            tracing::info!("🚀 Metal GPU detected, using acceleration for Llama-2-7B");
            match Device::new_metal(0) {
                Ok(device) => {
                    tracing::info!("✅ Metal device initialized successfully");
                    Ok(device)
                }
                Err(e) => {
                    tracing::warn!("❌ Metal initialization failed: {}, trying CUDA", e);
                    Self::try_cuda_or_cpu()
                }
            }
        } else if candle_core::utils::cuda_is_available() {
            tracing::info!("🚀 CUDA GPU detected, using acceleration for Llama-2-7B");
            Self::try_cuda_or_cpu()
        } else {
            tracing::info!("💾 No GPU detected, using CPU for Llama-2-7B (may be slow)");
            Ok(Device::Cpu)
        }
    }

    fn try_cuda_or_cpu() -> Result<Device> {
        match Device::new_cuda(0) {
            Ok(device) => {
                tracing::info!("✅ CUDA device initialized successfully");
                Ok(device)
            }
            Err(e) => {
                tracing::warn!("❌ CUDA initialization failed: {}, falling back to CPU", e);
                Ok(Device::Cpu)
            }
        }
    }

    async fn load_model_components(device: &Device) -> Result<(LlamaWrapper, Tokenizer, Config)> {
        let device_clone = device.clone();
        Self::load_model_async(device_clone).await
    }

    async fn load_model_async(device: Device) -> Result<(LlamaWrapper, Tokenizer, Config)> {
        tracing::info!("📥 Downloading Llama-2-7B-Chat from HuggingFace...");
        
        let mut api_builder = ApiBuilder::new();
        
        // Check for HuggingFace token (required for Llama-2)
        if let Ok(token) = env::var("HF_TOKEN") {
            tracing::info!("🔑 Using HuggingFace token for Llama-2 access");
            api_builder = api_builder.with_token(Some(token));
        } else {
            tracing::warn!("⚠️ No HF_TOKEN found - Llama-2 download may fail without authentication");
        }

        let api = api_builder.build()?;
        let repo = api.repo(Repo::with_revision(
            "meta-llama/Llama-2-7b-chat-hf".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        // Download configuration and tokenizer
        tracing::info!("📋 Downloading configuration files...");
        let config_file = repo.get("config.json").await
            .map_err(|e| anyhow::anyhow!("Failed to download config.json: {}", e))?;
        let tokenizer_file = repo.get("tokenizer.json").await
            .map_err(|e| anyhow::anyhow!("Failed to download tokenizer.json: {}", e))?;

        let config_json = std::fs::read_to_string(&config_file)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let config = Self::parse_llama2_config(&config_json)?;
        tracing::info!(
            "📊 Llama-2-7B Config: vocab_size={}, hidden_size={}, num_layers={}, context_length={}",
            config.vocab_size, config.hidden_size, config.num_hidden_layers, config.max_position_embeddings
        );

        // Download and load model weights
        tracing::info!("⬇️ Downloading Llama-2-7B weights (~13GB)...");
        let weight_files = Self::discover_weight_files(&repo).await?;
        tracing::info!("✅ Found {} weight file(s)", weight_files.len());

        let vars = Self::load_weights(&weight_files, &device)?;
        
        tracing::info!("🏗️ Building Llama-2-7B model graph...");
        let model = Llama::load(vars, &config)?;
        
        // Create cache with appropriate dtype
        let dtype = if matches!(device, Device::Cpu) { DType::F32 } else { DType::F16 };
        let cache = Cache::new(true, dtype, &config, &device)?;
        
        let wrapper = LlamaWrapper {
            model,
            cache,
            config: config.clone(),
            dtype,
        };

        tracing::info!("✅ Llama-2-7B model components loaded successfully");
        Ok((wrapper, tokenizer, config))
    }

    fn parse_llama2_config(config_json: &str) -> Result<Config> {
        let config: Value = serde_json::from_str(config_json)?;

        // Llama-2-7B specific configuration
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(4096) as usize;
        let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(11008) as usize;
        let num_hidden_layers = config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let num_attention_heads = config["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let num_key_value_heads = config
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(32); // Llama-2-7B uses full attention (no GQA)
        let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-5);
        let rope_theta = config
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let max_position_embeddings = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize; // Llama-2 4K context

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
                config.get("bos_token_id")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(1) as u32,
            ),
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(
                config.get("eos_token_id")
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

    async fn discover_weight_files(repo: &hf_hub::api::tokio::ApiRepo) -> Result<Vec<std::path::PathBuf>> {
        // Try different weight file patterns for Llama-2-7B
        let patterns = vec![
            // Single SafeTensors file (preferred)
            vec!["model.safetensors"],
            // Sharded SafeTensors files
            vec![
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
            vec![
                "model-00001-of-00003.safetensors", 
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            ],
            // PyTorch files as fallback
            vec!["pytorch_model.bin"],
            vec![
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
            ],
        ];

        for pattern in patterns {
            let mut files = Vec::new();
            let mut all_found = true;

            for filename in &pattern {
                match repo.get(filename).await {
                    Ok(path) => {
                        tracing::debug!("✅ Found weight file: {}", filename);
                        files.push(path);
                    }
                    Err(_) => {
                        all_found = false;
                        break;
                    }
                }
            }

            if all_found && !files.is_empty() {
                tracing::info!("📦 Using weight pattern: {:?}", pattern);
                return Ok(files);
            }
        }

        Err(anyhow::anyhow!("No compatible weight files found for Llama-2-7B"))
    }

    fn load_weights(weight_files: &[std::path::PathBuf], device: &Device) -> Result<VarBuilder<'static>> {
        use std::collections::HashMap;

        let mut all_tensors = HashMap::new();
        
        for (i, weight_file) in weight_files.iter().enumerate() {
            tracing::info!("📥 Loading weight file {}/{}: {:?}", i+1, weight_files.len(), weight_file.file_name());
            
            let tensors = if weight_file.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                candle_core::safetensors::load(weight_file, device)?
            } else {
                candle_core::pickle::read_all(weight_file)?
                    .into_iter()
                    .collect()
            };
            
            all_tensors.extend(tensors);
        }

        tracing::info!("✅ Loaded {} tensors total", all_tensors.len());
        
        // Use F16 for GPU, F32 for CPU
        let dtype = if matches!(device, Device::Cpu) { DType::F32 } else { DType::F16 };
        Ok(VarBuilder::from_tensors(all_tensors, dtype, device))
    }

    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let total_start = std::time::Instant::now();
        
        tracing::info!("🎯 [Llama2-7B] Starting generation for prompt: '{}' (max_tokens: {})", 
                      prompt.chars().take(50).collect::<String>(), max_tokens);
        
        // Apply Llama-2-Chat template
        let formatted_prompt = self.format_llama2_chat_template(prompt)?;
        tracing::info!("🎭 [Llama2-7B] Applied chat template, formatted length: {} chars", 
                      formatted_prompt.len());
        tracing::debug!("🎭 [Llama2-7B] Full formatted prompt: {}", formatted_prompt);
        
        // Tokenize
        let tokenize_start = std::time::Instant::now();
        tracing::info!("🔤 [Llama2-7B] Starting tokenization...");
        let tokens = self.tokenizer
            .encode(formatted_prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let input_ids = tokens.get_ids().to_vec();
        let tokenize_time = tokenize_start.elapsed();

        if input_ids.is_empty() {
            return Err(anyhow::anyhow!("Empty tokenization result"));
        }

        tracing::info!("🔤 [Llama2-7B] Tokenized {} tokens in {:?}", input_ids.len(), tokenize_time);
        tracing::debug!("🔤 [Llama2-7B] Token IDs: {:?}", &input_ids[..input_ids.len().min(10)]);

        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)?
            .reshape((1, input_ids.len()))?;

        // Reset cache for new conversation
        tracing::info!("🧹 [Llama2-7B] Resetting model cache for new conversation");
        
        let mut generated_tokens = Vec::with_capacity(max_tokens);
        let generation_start = std::time::Instant::now();
        
        // MAJOR FIX: Acquire model lock ONCE for entire generation process
        tracing::info!("🔒 [Llama2-7B] Acquiring model lock for entire generation process");
        let mut model_guard = self.model.lock().await;
        
        // Reset cache while holding the lock
        model_guard.reset_cache(&self.device)?;
        
        // Process initial prompt while holding the lock
        let mut seq_offset = 0;
        tracing::info!("🚀 [Llama2-7B] Running initial forward pass with {} input tokens", input_ids.len());
        let forward_start = std::time::Instant::now();
        let output = model_guard.forward(&input_tensor, seq_offset)?;
        let forward_time = forward_start.elapsed();
        tracing::info!("🚀 [Llama2-7B] Initial forward pass completed in {:?}", forward_time);
        
        seq_offset += input_ids.len();
        
        // Get last token logits
        let logits = output.squeeze(0)?;
        let last_logits = if logits.dims().len() > 1 {
            logits.narrow(0, logits.dim(0)? - 1, 1)?.squeeze(0)?
        } else {
            logits
        };
        
        // Sample first token with fast top-k sampling
        let bos_id = self.config.bos_token_id.unwrap_or(1) as u32;
        let top_k = 50; // Good balance of quality vs speed
        let mut next_token = Self::sample_token_topk_bos_temperature(&last_logits, bos_id, top_k, 0.8)?;
        
        // Generate remaining tokens - OPTIMIZED: Already holding lock
        tracing::info!("🔄 [Llama2-7B] Starting token-by-token generation loop (max {} tokens)", max_tokens);
        
        for step in 0..max_tokens {
            let loop_start = std::time::Instant::now();
            
            // Check for EOS
            if self.is_eos_token(next_token) {
                tracing::info!("🛑 [Llama2-7B] Hit EOS token {} at step {}", next_token, step);
                break;
            }
            
            generated_tokens.push(next_token);
            
            // Log progress every 10 tokens
            if step % 10 == 0 || step < 5 {
                tracing::info!("🔄 [Llama2-7B] Step {}: Generated token {} (total: {} tokens)", 
                              step, next_token, generated_tokens.len());
            }
            
            // Prepare next input - OPTIMIZED: Direct tensor creation
            let tensor_start = std::time::Instant::now();
            let next_input = Tensor::new(&[next_token], &self.device)?
                .reshape((1, 1))?;
            let tensor_time = tensor_start.elapsed();
            
            // Forward pass - NO ASYNC LOCK OVERHEAD
            let step_start = std::time::Instant::now();
            let output = model_guard.forward(&next_input, seq_offset)?;
            let step_time = step_start.elapsed();
            
            seq_offset += 1;
            
            // Sample next token - OPTIMIZED: Avoid double squeeze if possible
            let sample_start = std::time::Instant::now();
            let logits = if output.dims().len() > 2 {
                output.squeeze(0)?.squeeze(0)?
            } else if output.dims().len() > 1 {
                output.squeeze(0)?
            } else {
                output
            };
            next_token = Self::sample_token_topk_bos_temperature(&logits, bos_id, top_k, 0.8)?;
            let sample_time = sample_start.elapsed();
            
            let loop_time = loop_start.elapsed();
            
            if step < 5 || loop_time.as_millis() > 100 {
                tracing::info!("🚀 [Llama2-7B] Step {} timings: total={:?}, tensor={:?}, forward={:?}, sample={:?}", 
                              step, loop_time, tensor_time, step_time, sample_time);
            }
        }
        
        // Release the model lock after all generation is complete
        drop(model_guard);

        let generation_time = generation_start.elapsed();
        
        // Decode generated tokens
        let decode_start = std::time::Instant::now();
        let generated_text = self.tokenizer.decode(&generated_tokens, false)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        let decode_time = decode_start.elapsed();

        let total_time = total_start.elapsed();
        let tokens_per_second = generated_tokens.len() as f64 / generation_time.as_secs_f64();

        tracing::info!(
            "🚀 Llama-2-7B Performance: {} tokens in {:?} ({:.1} tok/s) | Tokenize: {:?} | Generate: {:?} | Decode: {:?}",
            generated_tokens.len(),
            total_time,
            tokens_per_second,
            tokenize_time,
            generation_time,
            decode_time
        );

        Ok(generated_text)
    }

    fn format_llama2_chat_template(&self, prompt: &str) -> Result<String> {
        // Official Llama-2-Chat template format
        // Reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        let formatted = format!(
            "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]",
            prompt
        );
        
        Ok(formatted)
    }
    
    /// Fast top-k sampling that bans BOS and samples from top_k candidates.
    /// Returns a token id (u32).
    // fn sample_token_topk_bos(logits: &Tensor, bos_id: u32, top_k: usize) -> Result<u32> {
    //     let sample_start = std::time::Instant::now();
        
    //     tracing::info!("🎲 [Llama2-7B] USING TOP-K SAMPLING with k={}, logits shape={:?}", top_k, logits.shape());
        
    //     // 1) OPTIMIZED: Only copy the logits tensor (should already be last token only)
    //     let cpu_start = std::time::Instant::now();
    //     let mut v = logits.to_vec1::<f32>()?;
    //     let vocab_size = v.len();
    //     let cpu_time = cpu_start.elapsed();
        
    //     tracing::info!("🎲 [Llama2-7B] CPU copy took {:?}, vocab_size={}", cpu_time, vocab_size);

    //     // 2) Ban BOS early
    //     if (bos_id as usize) < vocab_size {
    //         v[bos_id as usize] = f32::NEG_INFINITY;
    //     }

    //     // 3) SIMPLIFIED: Just sort and take top-k for now (we can optimize later)
    //     let k = top_k.min(vocab_size);
    //     let sort_start = std::time::Instant::now();
        
    //     // Create index-value pairs and sort by value descending
    //     let mut indexed_values: Vec<(usize, f32)> = v.iter().enumerate().map(|(i, &val)| (i, val)).collect();
    //     indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
    //     // Take top k
    //     indexed_values.truncate(k);
    //     let sort_time = sort_start.elapsed();
        
    //     tracing::info!("🎲 [Llama2-7B] Sort took {:?}", sort_time);

    //     // 4) Compute softmax weights on the selected top-k logits (stable softmax)
    //     let softmax_start = std::time::Instant::now();
    //     let mut max_logit = f32::NEG_INFINITY;
    //     for (_, val) in &indexed_values {
    //         if *val > max_logit { max_logit = *val; }
    //     }

    //     let mut weights = Vec::with_capacity(indexed_values.len());
    //     for (_, val) in &indexed_values {
    //         // subtract max for numerical stability
    //         weights.push((val - max_logit).exp());
    //     }
    //     let softmax_time = softmax_start.elapsed();
        
    //     tracing::info!("🎲 [Llama2-7B] Softmax took {:?}", softmax_time);

    //     // 5) Weighted sampling among top-k
    //     let sample_time_start = std::time::Instant::now();
    //     let total_weight: f32 = weights.iter().sum();
    //     let mut rng = rand::rng();
    //     let random_value = rng.random::<f32>() * total_weight;
        
    //     let mut cumulative_weight = 0.0;
    //     let mut choice = indexed_values[0].0; // fallback
        
    //     for (i, &weight) in weights.iter().enumerate() {
    //         cumulative_weight += weight;
    //         if random_value <= cumulative_weight {
    //             choice = indexed_values[i].0;
    //             break;
    //         }
    //     }
    //     let sample_time = sample_time_start.elapsed();

    //     let total_time = sample_start.elapsed();
    //     tracing::info!("🎲 [Llama2-7B] Top-k sampling COMPLETE: total={:?} (cpu={:?}, sort={:?}, softmax={:?}, sample={:?}), k={}, token={}", 
    //                    total_time, cpu_time, sort_time, softmax_time, sample_time, k, choice);
        
    //     Ok(choice as u32)
    // }


fn sample_token_topk_bos_temperature(logits: &Tensor, bos_id: u32, top_k: usize, temperature: f32) -> Result<u32> {
    let t_start = std::time::Instant::now();

    // ---- 1) Fast device argmax path (no full-vocab CPU copy) ----
    // Try argmax on device and convert to scalar. This transfers only a tiny scalar if on GPU.
    // If argmax != BOS, return immediately.
    if let Ok(dev_argmax) = logits.argmax(candle_core::D::Minus1) {
        if let Ok(candidate) = dev_argmax.to_scalar::<u32>() {
            if candidate != bos_id {
                tracing::debug!("🎲 Fast device argmax returned token {} (not BOS)", candidate);
                return Ok(candidate);
            }
            tracing::debug!("🎲 Fast device argmax returned BOS {}, falling back to CPU top-k", bos_id);
        } else {
            tracing::debug!("🎲 Fast device argmax to_scalar failed; falling back to CPU top-k");
        }
    } else {
        tracing::debug!("🎲 Fast device argmax failed; falling back to CPU top-k");
    }

    // ---- 2) Fallback: copy logits to CPU and apply temperature ----
    // IMPORTANT: Ensure `logits` is already the vector of shape [vocab] (last position).
    // If it's not, slice it to last position before calling this function.
    let cpu_start = std::time::Instant::now();
    let mut v = logits.to_vec1::<f32>()?; // single small copy of length ~= vocab_size
    
    // Apply temperature scaling (temperature < 1.0 = more focused, > 1.0 = more random)
    if temperature != 1.0 {
        for val in &mut v {
            *val /= temperature;
        }
    }
    
    let cpu_time = cpu_start.elapsed();

    let vocab_size = v.len();
    if vocab_size == 0 {
        return Err(anyhow::anyhow!("Empty logits on CPU fallback"));
    }

    // Ban BOS token
    if (bos_id as usize) < vocab_size {
        v[bos_id as usize] = f32::NEG_INFINITY;
    }

    // ---- 3) Efficient top-k selection (no full-sort if k << vocab) ----
    let k = top_k.max(1).min(vocab_size);
    // We'll get indices of top-k by using select_nth_unstable-like logic on index array.
    let mut idx: Vec<usize> = (0..vocab_size).collect();

    if k < vocab_size {
        // Keep the top k largest elements in idx[..k] (unordered)
        // Use partial comparator that compares v[*]
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            // descending: compare v[b] to v[a]
            v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
    }
    // If k == vocab_size, idx is full vocab.

    // ---- 4) Compute softmax weights for the selected top-k (stable) ----
    let mut max_logit = f32::NEG_INFINITY;
    for &i in &idx {
        if v[i] > max_logit { max_logit = v[i]; }
    }

    // Build weights vec (exp(logit - max))
    let mut weights: Vec<f32> = Vec::with_capacity(idx.len());
    for &i in &idx {
        weights.push((v[i] - max_logit).exp());
    }

    // ---- 5) Weighted sampling among top-k using WeightedIndex ----
    // Use rand's WeightedIndex which is well-optimized.
    let sampler_start = std::time::Instant::now();
    let dist = rand::distr::weighted::WeightedIndex::new(&weights)
        .map_err(|e| anyhow::anyhow!("WeightedIndex creation failed: {}", e))?;
    let mut rng = rand::rng();
    let sample_choice_local = dist.sample(&mut rng);
    let choice = idx[sample_choice_local];
    let sampler_time = sampler_start.elapsed();

    let total_time = t_start.elapsed();
    tracing::debug!(
        "🎲 Top-k sampling: total={:?}, cpu_copy={:?}, k={}, choice={}, sampler={:?}",
        total_time, cpu_time, idx.len(), choice, sampler_time
    );

    Ok(choice as u32)
}

    fn is_eos_token(&self, token: u32) -> bool {
        // Use config for EOS token - more robust than hardcoding
        if let Some(candle_transformers::models::llama::LlamaEosToks::Single(id)) = self.config.eos_token_id {
            token == id as u32
        } else {
            token == 2 // fallback to standard EOS
        }
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "Llama-2-7B-Chat".to_string(),
            version: "main".to_string(),
            parameters: 7_000_000_000, // 7B parameters
            memory_mb: self.estimate_memory_usage(),
            device: format!("{:?}", self.device),
            vocab_size: self.config.vocab_size,
            context_length: self.config.max_position_embeddings,
            model_type: "llama2-chat".to_string(),
            architecture: "transformer".to_string(),
            precision: if matches!(self.device, Device::Cpu) { "f32" } else { "f16" }.to_string(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        let params = 7_000_000_000; // 7B parameters
        let bytes_per_param = if matches!(self.device, Device::Cpu) { 4 } else { 2 }; // F32 vs F16
        let param_memory = params * bytes_per_param;
        let kv_cache_memory = 512 * 1024 * 1024; // ~512MB for KV cache
        let activation_memory = 256 * 1024 * 1024; // ~256MB for activations
        
        (param_memory + kv_cache_memory + activation_memory) / (1024 * 1024)
    }
}

// Implement ModelTrait for compatibility with the system
#[async_trait]
impl ModelTrait for Llama2ChatModel {
    async fn load() -> Result<BoxedModel>
    where
        Self: Sized,
    {
        let model = Llama2ChatModel::load().await?;
        Ok(Box::new(model))
    }

    async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate(prompt, max_tokens).await
    }

    fn model_info(&self) -> ModelInfo {
        self.model_info()
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn model_name(&self) -> &str {
        "llama2-7b-chat"
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "generation" => true,
            "health_check" => true,
            "chat_optimized" => true,
            "conversation" => true,
            "large_context" => true,
            "auth_required" => true, // Requires HF token
            _ => false,
        }
    }
}

// Register Llama-2-7B-Chat model with the registry
pub fn register_llama2_chat_model() -> Result<()> {
    use crate::models::registry::{global_registry, ModelRegistration};
    use std::sync::Arc;

    let registration = ModelRegistration {
        name: "llama2-7b-chat".to_string(),
        aliases: vec![
            "llama2".to_string(),
            "llama-2".to_string(),
            "llama2-chat".to_string(),
            "llama-2-chat".to_string(),
            "llama2-7b".to_string(),
            "meta-llama2".to_string(),
        ],
        description: "Meta Llama-2-7B-Chat - Production conversational AI model with official chat template".to_string(),
        model_type: "llama2-chat".to_string(),
        supported_features: vec![
            "generation".to_string(),
            "health_check".to_string(),
            "chat_optimized".to_string(),
            "conversation".to_string(),
            "large_context".to_string(),
            "auth_required".to_string(),
        ],
        memory_requirements_mb: 14000, // ~14GB total
        factory: Arc::new(Box::new(|| {
            Box::new(Box::pin(async {
                let model = Llama2ChatModel::load().await?;
                Ok(Box::new(model) as BoxedModel)
            }))
        })),
    };

    let alias_count = registration.aliases.len();
    global_registry().register_model(registration)?;
    tracing::info!("📋 Registered model: llama2-7b-chat with {} aliases", alias_count);
    Ok(())
}