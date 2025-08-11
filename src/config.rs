use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub models: ModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub cache_dir: String,
    pub max_concurrent_requests: usize,
    pub default_model: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: env::var("PORT")
                    .unwrap_or_else(|_| "8080".to_string())
                    .parse()
                    .unwrap_or(8080),
                log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            },
            models: ModelConfig {
                cache_dir: env::var("MODEL_CACHE_DIR")
                    .unwrap_or_else(|_| "./models".to_string()),
                max_concurrent_requests: env::var("MAX_CONCURRENT_REQUESTS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .unwrap_or(10),
                default_model: env::var("DEFAULT_MODEL")
                    .unwrap_or_else(|_| "microsoft/DialoGPT-medium".to_string()),
            },
        }
    }
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        Ok(Self::default())
    }
}