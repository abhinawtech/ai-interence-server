// ARCHITECTURE: Dynamic Model Registry System
//
// DESIGN PHILOSOPHY:
// This registry enables runtime model discovery and loading without hardcoded dependencies.
// Models can be registered at startup, allowing for plugin-like architecture.
//
// KEY FEATURES:
// 1. RUNTIME REGISTRATION: Models register themselves at startup
// 2. FACTORY PATTERN: Each model provides a factory function for creation
// 3. ALIAS SUPPORT: Multiple names can point to the same model
// 4. METADATA STORAGE: Rich information about available models
// 5. THREAD SAFETY: Safe for concurrent access across the application

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use crate::models::traits::{BoxedModel, ModelTrait, ModelInfo};

/// Factory function for creating model instances
type ModelFactory = Box<dyn Fn() -> Box<dyn std::future::Future<Output = Result<BoxedModel>> + Send + Unpin> + Send + Sync>;

/// Registration information for a model
#[derive(Clone)]
pub struct ModelRegistration {
    pub name: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub model_type: String,
    pub supported_features: Vec<String>,
    pub memory_requirements_mb: usize,
    pub factory: Arc<ModelFactory>,
}

/// Thread-safe dynamic model registry
#[derive(Clone)]
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<String, ModelRegistration>>>,
    aliases: Arc<RwLock<HashMap<String, String>>>, // alias -> canonical_name mapping
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    /// Create a new empty model registry
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            aliases: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new model with the registry
    /// 
    /// # Arguments
    /// * `registration` - Model registration information including factory function
    pub fn register_model(&self, registration: ModelRegistration) -> Result<()> {
        let mut models = self.models.write().unwrap();
        let mut aliases = self.aliases.write().unwrap();

        // Register the main model name
        let name = registration.name.clone();
        models.insert(name.clone(), registration.clone());

        // Register all aliases
        for alias in &registration.aliases {
            aliases.insert(alias.clone(), name.clone());
        }

        tracing::info!("📋 Registered model: {} with {} aliases", name, registration.aliases.len());
        Ok(())
    }

    /// Load a model by name or alias
    /// 
    /// # Arguments
    /// * `name` - Model name or alias to load
    /// 
    /// # Returns
    /// Boxed model trait object
    pub async fn load_model(&self, name: &str) -> Result<BoxedModel> {
        let canonical_name = self.resolve_name(name)?;
        let registration = {
            let models = self.models.read().unwrap();
            models.get(&canonical_name)
                .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in registry", canonical_name))?
                .clone()
        };

        tracing::info!("🔄 Loading model: {} (requested as: {})", canonical_name, name);
        
        // Create factory future and await it
        let factory = registration.factory;
        let future = factory();
        let model = future.await?;

        tracing::info!("✅ Successfully loaded model: {}", canonical_name);
        Ok(model)
    }

    /// Resolve a name or alias to canonical model name
    fn resolve_name(&self, name: &str) -> Result<String> {
        // First check if it's a canonical name
        {
            let models = self.models.read().unwrap();
            if models.contains_key(name) {
                return Ok(name.to_string());
            }
        }

        // Then check aliases
        {
            let aliases = self.aliases.read().unwrap();
            if let Some(canonical) = aliases.get(name) {
                return Ok(canonical.clone());
            }
        }

        Err(anyhow::anyhow!("Model '{}' not found in registry", name))
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<ModelRegistration> {
        let models = self.models.read().unwrap();
        models.values().cloned().collect()
    }

    /// Get model registration by name
    pub fn get_registration(&self, name: &str) -> Result<ModelRegistration> {
        let canonical_name = self.resolve_name(name)?;
        let models = self.models.read().unwrap();
        models.get(&canonical_name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", canonical_name))
    }

    /// Check if a model is registered
    pub fn is_registered(&self, name: &str) -> bool {
        self.resolve_name(name).is_ok()
    }

    /// Get all aliases for a model
    pub fn get_aliases(&self, name: &str) -> Result<Vec<String>> {
        let canonical_name = self.resolve_name(name)?;
        let models = self.models.read().unwrap();
        if let Some(registration) = models.get(&canonical_name) {
            Ok(registration.aliases.clone())
        } else {
            Ok(vec![])
        }
    }
}

/// Global static registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<ModelRegistry> = std::sync::OnceLock::new();

/// Get the global model registry instance
pub fn global_registry() -> &'static ModelRegistry {
    GLOBAL_REGISTRY.get_or_init(ModelRegistry::new)
}

/// Initialize the global registry with default models
pub fn initialize_registry() {
    let registry = global_registry();
    tracing::info!("🏭 Initializing global model registry...");
    
    // Models will register themselves in their respective modules
    // This function just ensures the registry is initialized
}

/// Helper macro for easy model registration
#[macro_export]
macro_rules! register_model {
    ($name:expr, $factory:expr, $aliases:expr, $description:expr, $model_type:expr) => {
        {
            use $crate::models::registry::{global_registry, ModelRegistration};
            use std::sync::Arc;
            
            let registration = ModelRegistration {
                name: $name.to_string(),
                aliases: $aliases.into_iter().map(|s| s.to_string()).collect(),
                description: $description.to_string(),
                model_type: $model_type.to_string(),
                supported_features: vec!["generation".to_string(), "health_check".to_string()],
                memory_requirements_mb: 2048, // Default, should be overridden
                factory: Arc::new(Box::new(|| {
                    Box::new($factory()) as Box<dyn std::future::Future<Output = anyhow::Result<Box<dyn $crate::models::traits::ModelTrait>>> + Send + Unpin>
                })),
            };
            
            global_registry().register_model(registration)
        }
    };
}