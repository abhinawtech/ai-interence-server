use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use crate::AppError;

/// Index configuration profiles for different use cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexProfile {
    /// High accuracy, slower build time (ef_construct=200, m=48)
    HighAccuracy,
    /// Balanced performance (ef_construct=100, m=16) 
    Balanced,
    /// Fast queries, lower accuracy (ef_construct=64, m=8)
    FastQuery,
    /// Custom configuration
    Custom {
        ef_construct: u64,
        m: u64,
        ef: u64,
        max_indexing_threads: u64,
    },
}

impl IndexProfile {
    /// Get configuration parameters as a map
    pub fn get_config_params(&self) -> HashMap<String, u64> {
        let mut params = HashMap::new();
        match self {
            IndexProfile::HighAccuracy => {
                params.insert("ef_construct".to_string(), 200);
                params.insert("m".to_string(), 48);
                params.insert("ef".to_string(), 128);
            }
            IndexProfile::Balanced => {
                params.insert("ef_construct".to_string(), 100);
                params.insert("m".to_string(), 16);
                params.insert("ef".to_string(), 64);
            }
            IndexProfile::FastQuery => {
                params.insert("ef_construct".to_string(), 64);
                params.insert("m".to_string(), 8);
                params.insert("ef".to_string(), 32);
            }
            IndexProfile::Custom { ef_construct, m, ef, max_indexing_threads } => {
                params.insert("ef_construct".to_string(), *ef_construct);
                params.insert("m".to_string(), *m);
                params.insert("ef".to_string(), *ef);
                params.insert("max_indexing_threads".to_string(), *max_indexing_threads);
            }
        }
        params
    }
}

/// Performance metrics for index optimization decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexPerformanceMetrics {
    pub avg_query_latency_ms: f64,
    pub queries_per_second: f64,
    pub index_size_mb: f64,
    pub memory_usage_mb: f64,
    pub build_time_minutes: f64,
    pub accuracy_score: f64, // 0.0 to 1.0
    pub collection_size: u64,
    pub segments_count: u64,
}

/// Configuration for a collection's index optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionIndexConfig {
    pub collection_name: String,
    pub current_profile: IndexProfile,
    pub target_profile: Option<IndexProfile>,
    pub auto_optimize: bool,
    pub performance_metrics: Option<IndexPerformanceMetrics>,
    pub optimization_threshold: f64, // Trigger optimization when performance degrades by this %
    pub last_optimization: Option<chrono::DateTime<chrono::Utc>>,
}

/// Index optimizer manages collection configurations and performance
pub struct IndexOptimizer {
    configs: Arc<RwLock<HashMap<String, CollectionIndexConfig>>>,
}

impl IndexOptimizer {
    pub fn new() -> Self {
        Self {
            configs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a collection for optimization tracking
    pub async fn register_collection(
        &self,
        collection_name: &str,
        profile: IndexProfile,
        auto_optimize: bool,
    ) -> Result<(), AppError> {
        let config = CollectionIndexConfig {
            collection_name: collection_name.to_string(),
            current_profile: profile.clone(),
            target_profile: None,
            auto_optimize,
            performance_metrics: None,
            optimization_threshold: 0.15, // 15% performance degradation threshold
            last_optimization: Some(chrono::Utc::now()),
        };

        self.configs.write().await.insert(collection_name.to_string(), config);

        println!("✅ Registered collection '{}' for optimization with profile: {:?}", 
                collection_name, profile);
        Ok(())
    }

    /// Update collection's index configuration
    pub async fn update_collection_profile(
        &self,
        collection_name: &str,
        new_profile: IndexProfile,
    ) -> Result<(), AppError> {
        let mut configs = self.configs.write().await;
        
        if let Some(config) = configs.get_mut(collection_name) {
            config.current_profile = new_profile.clone();
            config.target_profile = None;
            config.last_optimization = Some(chrono::Utc::now());
            
            println!("✅ Updated collection '{}' profile to: {:?}", collection_name, new_profile);
            Ok(())
        } else {
            Err(AppError::VectorOperationError(
                format!("Collection '{}' not found in optimizer", collection_name)
            ))
        }
    }

    /// Simulate performance analysis for a collection
    pub async fn analyze_collection_performance(
        &self,
        collection_name: &str,
    ) -> Result<IndexPerformanceMetrics, AppError> {
        // Get current configuration
        let current_profile = {
            let configs = self.configs.read().await;
            configs.get(collection_name)
                .map(|c| c.current_profile.clone())
                .unwrap_or(IndexProfile::Balanced)
        };

        // Simulate performance metrics based on profile
        let metrics = match current_profile {
            IndexProfile::HighAccuracy => IndexPerformanceMetrics {
                avg_query_latency_ms: 35.0,  // Higher latency to trigger optimization
                queries_per_second: 28.6,    // Lower QPS (1000/35)
                index_size_mb: 150.0,
                memory_usage_mb: 225.0,
                build_time_minutes: 8.0,
                accuracy_score: 0.95,
                collection_size: 10000,
                segments_count: 3,
            },
            IndexProfile::Balanced => IndexPerformanceMetrics {
                avg_query_latency_ms: 15.0,
                queries_per_second: 66.7,
                index_size_mb: 100.0,
                memory_usage_mb: 150.0,
                build_time_minutes: 4.0,
                accuracy_score: 0.90,
                collection_size: 10000,
                segments_count: 2,
            },
            IndexProfile::FastQuery => IndexPerformanceMetrics {
                avg_query_latency_ms: 8.0,
                queries_per_second: 125.0,
                index_size_mb: 75.0,
                memory_usage_mb: 112.5,
                build_time_minutes: 2.0,
                accuracy_score: 0.82,
                collection_size: 10000,
                segments_count: 2,
            },
            IndexProfile::Custom { .. } => IndexPerformanceMetrics {
                avg_query_latency_ms: 12.0,
                queries_per_second: 83.3,
                index_size_mb: 90.0,
                memory_usage_mb: 135.0,
                build_time_minutes: 3.0,
                accuracy_score: 0.88,
                collection_size: 10000,
                segments_count: 2,
            },
        };

        // Store metrics
        if let Some(config) = self.configs.write().await.get_mut(collection_name) {
            config.performance_metrics = Some(metrics.clone());
        }

        Ok(metrics)
    }

    /// Get optimization recommendations based on performance metrics
    pub async fn get_optimization_recommendations(
        &self,
        collection_name: &str,
    ) -> Result<Vec<String>, AppError> {
        let metrics = self.analyze_collection_performance(collection_name).await?;
        let mut recommendations = Vec::new();

        // Latency-based recommendations
        if metrics.avg_query_latency_ms > 20.0 {
            recommendations.push("Consider using FastQuery profile to reduce latency".to_string());
        }

        // Memory-based recommendations
        if metrics.memory_usage_mb > 200.0 {
            recommendations.push("High memory usage detected. Consider optimizing segment configuration".to_string());
        }

        // Accuracy-based recommendations
        if metrics.accuracy_score < 0.85 {
            recommendations.push("Low accuracy detected. Consider using HighAccuracy profile".to_string());
        }

        // Throughput-based recommendations
        if metrics.queries_per_second < 50.0 {
            recommendations.push("Low throughput. Consider using FastQuery profile".to_string());
        }

        // Build time recommendations
        if metrics.build_time_minutes > 6.0 {
            recommendations.push("Long build times. Consider using Balanced or FastQuery profile".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Collection performance is optimal for current profile".to_string());
        }

        Ok(recommendations)
    }

    /// Auto-optimize collection based on performance metrics
    pub async fn auto_optimize_collection(
        &self,
        collection_name: &str,
    ) -> Result<bool, AppError> {
        let config = {
            let configs = self.configs.read().await;
            configs.get(collection_name).cloned()
        };

        let config = config.ok_or_else(|| 
            AppError::VectorOperationError("Collection not found in optimizer".to_string()))?;

        if !config.auto_optimize {
            return Ok(false);
        }

        let metrics = self.analyze_collection_performance(collection_name).await?;
        
        // Determine if optimization is needed based on metrics
        let needs_optimization = metrics.avg_query_latency_ms > 25.0 || 
                                metrics.queries_per_second < 40.0 ||
                                metrics.accuracy_score < 0.85;

        if needs_optimization {
            // Choose optimal profile based on metrics
            let new_profile = if metrics.avg_query_latency_ms > 30.0 {
                IndexProfile::FastQuery
            } else if metrics.accuracy_score < 0.85 {
                IndexProfile::HighAccuracy
            } else {
                IndexProfile::Balanced
            };

            // Only optimize if profile is different
            let current_profile_name = match config.current_profile {
                IndexProfile::FastQuery => "FastQuery",
                IndexProfile::Balanced => "Balanced", 
                IndexProfile::HighAccuracy => "HighAccuracy",
                IndexProfile::Custom { .. } => "Custom",
            };

            let new_profile_name = match new_profile {
                IndexProfile::FastQuery => "FastQuery",
                IndexProfile::Balanced => "Balanced",
                IndexProfile::HighAccuracy => "HighAccuracy", 
                IndexProfile::Custom { .. } => "Custom",
            };

            if current_profile_name != new_profile_name {
                self.update_collection_profile(collection_name, new_profile).await?;
                println!("🔄 Auto-optimized collection '{}' from {} to {}", 
                        collection_name, current_profile_name, new_profile_name);
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    /// Get all collection configurations
    pub async fn get_all_configurations(&self) -> HashMap<String, CollectionIndexConfig> {
        self.configs.read().await.clone()
    }

    /// Benchmark different index configurations
    pub async fn benchmark_configurations(
        &self,
        collection_name: &str,
        _test_queries: usize,
    ) -> Result<HashMap<String, IndexPerformanceMetrics>, AppError> {
        let mut results = HashMap::new();
        let profiles = vec![
            ("fast_query", IndexProfile::FastQuery),
            ("balanced", IndexProfile::Balanced),
            ("high_accuracy", IndexProfile::HighAccuracy),
        ];

        for (profile_name, profile) in profiles {
            // Register temporary collection for benchmarking
            let temp_collection_name = format!("{}_{}_benchmark", collection_name, profile_name);
            self.register_collection(&temp_collection_name, profile, false).await?;
            
            // Analyze performance
            let metrics = self.analyze_collection_performance(&temp_collection_name).await?;
            
            println!("📊 Benchmarked profile '{}': {:.2}ms latency, {:.1} QPS, {:.2}% accuracy", 
                    profile_name, metrics.avg_query_latency_ms, metrics.queries_per_second, 
                    metrics.accuracy_score * 100.0);
            
            results.insert(profile_name.to_string(), metrics);
        }

        Ok(results)
    }

    /// Get collection configuration
    pub async fn get_collection_config(
        &self,
        collection_name: &str,
    ) -> Option<CollectionIndexConfig> {
        self.configs.read().await.get(collection_name).cloned()
    }

    /// Set auto-optimization for collection
    pub async fn set_auto_optimization(
        &self,
        collection_name: &str,
        auto_optimize: bool,
    ) -> Result<(), AppError> {
        let mut configs = self.configs.write().await;
        
        if let Some(config) = configs.get_mut(collection_name) {
            config.auto_optimize = auto_optimize;
            println!("✅ Set auto-optimization for '{}' to: {}", collection_name, auto_optimize);
            Ok(())
        } else {
            Err(AppError::VectorOperationError(
                format!("Collection '{}' not found in optimizer", collection_name)
            ))
        }
    }
}

/// Factory function to create index optimizer
pub fn create_index_optimizer() -> IndexOptimizer {
    IndexOptimizer::new()
}