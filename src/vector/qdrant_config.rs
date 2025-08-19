// ================================================================================================
// QDRANT CONFIGURATION - PRODUCTION READY VECTOR DATABASE SETUP
// ================================================================================================
//
// Environment-based configuration for different deployment scenarios:
// - Development: Local Qdrant instance
// - Staging: Docker Compose setup
// - Production: Kubernetes/Cloud deployment
//
// ================================================================================================

use serde::{Deserialize, Serialize};
use std::env;

/// Qdrant client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    /// Qdrant server URL (e.g., "http://localhost:6334")
    pub url: String,
    
    /// API key for authentication (optional)
    pub api_key: Option<String>,
    
    /// Connection timeout in seconds
    pub timeout_secs: u64,
    
    /// Maximum number of connections in pool
    pub max_connections: usize,
    
    /// Vector dimension for embeddings
    pub vector_dimension: usize,
    
    /// Collection name for conversations
    pub conversation_collection: String,
    
    /// Enable TLS/SSL
    pub use_tls: bool,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            api_key: None,
            timeout_secs: 30,
            max_connections: 10,
            vector_dimension: 64,
            conversation_collection: "conversations".to_string(),
            use_tls: false,
        }
    }
}

impl QdrantConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        Self {
            url: env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6334".to_string()),
            
            api_key: env::var("QDRANT_API_KEY").ok(),
            
            timeout_secs: env::var("QDRANT_TIMEOUT_SECS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            
            max_connections: env::var("QDRANT_MAX_CONNECTIONS")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .unwrap_or(10),
            
            vector_dimension: env::var("QDRANT_VECTOR_DIMENSION")
                .unwrap_or_else(|_| "64".to_string())
                .parse()
                .unwrap_or(64),
            
            conversation_collection: env::var("QDRANT_CONVERSATION_COLLECTION")
                .unwrap_or_else(|_| "conversations".to_string()),
            
            use_tls: env::var("QDRANT_USE_TLS")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.url.is_empty() {
            return Err("Qdrant URL cannot be empty".to_string());
        }
        
        if self.timeout_secs == 0 {
            return Err("Timeout must be greater than 0".to_string());
        }
        
        if self.max_connections == 0 {
            return Err("Max connections must be greater than 0".to_string());
        }
        
        if self.vector_dimension == 0 {
            return Err("Vector dimension must be greater than 0".to_string());
        }
        
        if self.conversation_collection.is_empty() {
            return Err("Collection name cannot be empty".to_string());
        }
        
        Ok(())
    }
    
    /// Get configuration for development environment
    pub fn development() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            api_key: None,
            timeout_secs: 10,
            max_connections: 5,
            vector_dimension: 64,
            conversation_collection: "conversations_dev".to_string(),
            use_tls: false,
        }
    }
    
    /// Get configuration for production environment
    pub fn production() -> Self {
        Self {
            url: env::var("QDRANT_URL")
                .expect("QDRANT_URL must be set in production"),
            api_key: env::var("QDRANT_API_KEY").ok(),
            timeout_secs: 30,
            max_connections: 20,
            vector_dimension: 64,
            conversation_collection: "conversations".to_string(),
            use_tls: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = QdrantConfig::default();
        assert_eq!(config.url, "http://localhost:6334");
        assert_eq!(config.vector_dimension, 64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = QdrantConfig::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Empty URL should fail
        config.url = "".to_string();
        assert!(config.validate().is_err());
        
        // Zero timeout should fail
        config.url = "http://localhost:6334".to_string();
        config.timeout_secs = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_development_config() {
        let config = QdrantConfig::development();
        assert_eq!(config.conversation_collection, "conversations_dev");
        assert_eq!(config.max_connections, 5);
        assert!(!config.use_tls);
    }
}