use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;
use std::sync::Arc;
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};
use crate::AppError;

/// Time-series metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

/// Performance metrics for a specific time window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceWindow {
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub avg_query_latency_ms: f64,
    pub p95_query_latency_ms: f64,
    pub p99_query_latency_ms: f64,
    pub queries_per_second: f64,
    pub error_rate_percent: f64,
    pub index_size_mb: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub cache_hit_rate_percent: f64,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert condition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub rule_id: Uuid,
    pub rule_name: String,
    pub metric_name: String,
    pub threshold: f64,
    pub comparison: AlertComparison,
    pub severity: AlertSeverity,
    pub duration_minutes: u64, // How long condition must persist
    pub enabled: bool,
    pub collection_pattern: Option<String>, // Regex pattern for collection names
}

/// Alert comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertComparison {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Active alert instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub alert_id: Uuid,
    pub rule: AlertRule,
    pub collection_name: String,
    pub triggered_at: DateTime<Utc>,
    pub current_value: f64,
    pub message: String,
    pub resolved_at: Option<DateTime<Utc>>,
}

/// Health status for a collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Collection health summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionHealth {
    pub collection_name: String,
    pub status: HealthStatus,
    pub last_check: DateTime<Utc>,
    pub active_alerts: Vec<ActiveAlert>,
    pub performance_score: f64, // 0.0 to 1.0
    pub recommendations: Vec<String>,
}

/// Time series data storage
#[derive(Debug)]
pub struct TimeSeriesStorage {
    // Collection name -> Metric name -> Time-ordered data points
    data: HashMap<String, HashMap<String, VecDeque<MetricDataPoint>>>,
    max_retention_hours: u64,
}

impl TimeSeriesStorage {
    pub fn new(max_retention_hours: u64) -> Self {
        Self {
            data: HashMap::new(),
            max_retention_hours,
        }
    }

    /// Add a metric data point
    pub fn add_metric(&mut self, collection: &str, metric: &str, value: f64, tags: HashMap<String, String>) {
        let timestamp = Utc::now();
        let data_point = MetricDataPoint {
            timestamp,
            value,
            tags,
        };

        self.data
            .entry(collection.to_string())
            .or_insert_with(HashMap::new)
            .entry(metric.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(data_point);

        // Clean old data
        self.cleanup_old_data(collection, metric);
    }

    /// Get metric history for a collection
    pub fn get_metric_history(&self, collection: &str, metric: &str, hours: u64) -> Vec<MetricDataPoint> {
        let cutoff = Utc::now() - Duration::hours(hours as i64);
        
        self.data
            .get(collection)
            .and_then(|metrics| metrics.get(metric))
            .map(|points| {
                points
                    .iter()
                    .filter(|point| point.timestamp >= cutoff)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Calculate performance window from metrics
    pub fn calculate_performance_window(&self, collection: &str, window_minutes: u64) -> Option<PerformanceWindow> {
        let window_start = Utc::now() - Duration::minutes(window_minutes as i64);
        let window_end = Utc::now();

        let latency_points = self.get_metric_history(collection, "query_latency_ms", 1);
        let qps_points = self.get_metric_history(collection, "queries_per_second", 1);
        let error_points = self.get_metric_history(collection, "error_rate", 1);
        let size_points = self.get_metric_history(collection, "index_size_mb", 1);
        let memory_points = self.get_metric_history(collection, "memory_usage_mb", 1);
        let cpu_points = self.get_metric_history(collection, "cpu_usage_percent", 1);
        let cache_points = self.get_metric_history(collection, "cache_hit_rate", 1);

        if latency_points.is_empty() {
            return None;
        }

        // Calculate statistics
        let mut latencies: Vec<f64> = latency_points.iter().map(|p| p.value).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p95_latency = Self::percentile(&latencies, 0.95);
        let p99_latency = Self::percentile(&latencies, 0.99);

        let avg_qps = qps_points.iter().map(|p| p.value).sum::<f64>() / qps_points.len().max(1) as f64;
        let avg_error_rate = error_points.iter().map(|p| p.value).sum::<f64>() / error_points.len().max(1) as f64;
        let latest_size = size_points.last().map(|p| p.value).unwrap_or(0.0);
        let avg_memory = memory_points.iter().map(|p| p.value).sum::<f64>() / memory_points.len().max(1) as f64;
        let avg_cpu = cpu_points.iter().map(|p| p.value).sum::<f64>() / cpu_points.len().max(1) as f64;
        let avg_cache_hit = cache_points.iter().map(|p| p.value).sum::<f64>() / cache_points.len().max(1) as f64;

        Some(PerformanceWindow {
            window_start,
            window_end,
            avg_query_latency_ms: avg_latency,
            p95_query_latency_ms: p95_latency,
            p99_query_latency_ms: p99_latency,
            queries_per_second: avg_qps,
            error_rate_percent: avg_error_rate,
            index_size_mb: latest_size,
            memory_usage_mb: avg_memory,
            cpu_usage_percent: avg_cpu,
            cache_hit_rate_percent: avg_cache_hit,
        })
    }

    fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }
        let index = (percentile * (sorted_values.len() - 1) as f64).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    fn cleanup_old_data(&mut self, collection: &str, metric: &str) {
        let cutoff = Utc::now() - Duration::hours(self.max_retention_hours as i64);
        
        if let Some(collection_data) = self.data.get_mut(collection) {
            if let Some(metric_data) = collection_data.get_mut(metric) {
                while let Some(front) = metric_data.front() {
                    if front.timestamp < cutoff {
                        metric_data.pop_front();
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

/// Index monitoring system
pub struct IndexMonitor {
    time_series: Arc<RwLock<TimeSeriesStorage>>,
    alert_rules: Arc<RwLock<HashMap<Uuid, AlertRule>>>,
    active_alerts: Arc<RwLock<HashMap<Uuid, ActiveAlert>>>,
    collection_health: Arc<RwLock<HashMap<String, CollectionHealth>>>,
}

impl IndexMonitor {
    pub fn new(retention_hours: u64) -> Self {
        Self {
            time_series: Arc::new(RwLock::new(TimeSeriesStorage::new(retention_hours))),
            alert_rules: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            collection_health: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a metric value
    pub async fn record_metric(&self, collection: &str, metric: &str, value: f64, tags: Option<HashMap<String, String>>) {
        let tags = tags.unwrap_or_default();
        self.time_series.write().await.add_metric(collection, metric, value, tags);
        
        // Check alert rules
        self.check_alerts(collection, metric, value).await;
        
        // Update collection health
        self.update_collection_health(collection).await;
    }

    /// Record query performance metrics
    pub async fn record_query_performance(&self, collection: &str, latency_ms: f64, success: bool) {
        let mut tags = HashMap::new();
        tags.insert("operation".to_string(), "query".to_string());
        
        self.record_metric(collection, "query_latency_ms", latency_ms, Some(tags.clone())).await;
        
        // Update QPS
        self.record_metric(collection, "queries_per_second", 1.0, Some(tags.clone())).await;
        
        // Update error rate
        let error_value = if success { 0.0 } else { 1.0 };
        self.record_metric(collection, "error_rate", error_value, Some(tags)).await;
        
        println!("📊 Recorded query performance: {}ms latency, success: {}", latency_ms, success);
    }

    /// Record system resource metrics
    pub async fn record_system_metrics(&self, collection: &str, cpu_percent: f64, memory_mb: f64, index_size_mb: f64) {
        let mut tags = HashMap::new();
        tags.insert("source".to_string(), "system".to_string());
        
        self.record_metric(collection, "cpu_usage_percent", cpu_percent, Some(tags.clone())).await;
        self.record_metric(collection, "memory_usage_mb", memory_mb, Some(tags.clone())).await;
        self.record_metric(collection, "index_size_mb", index_size_mb, Some(tags)).await;
    }

    /// Create an alert rule
    pub async fn create_alert_rule(&self, rule: AlertRule) -> Result<(), AppError> {
        let rule_id = rule.rule_id;
        self.alert_rules.write().await.insert(rule_id, rule);
        println!("🚨 Created alert rule: {}", rule_id);
        Ok(())
    }

    /// Check alert conditions
    async fn check_alerts(&self, collection: &str, metric: &str, value: f64) {
        let rules = self.alert_rules.read().await;
        
        for rule in rules.values() {
            if !rule.enabled || rule.metric_name != metric {
                continue;
            }
            
            // Check collection pattern if specified
            if let Some(pattern) = &rule.collection_pattern {
                if !collection.contains(pattern) {
                    continue;
                }
            }
            
            let triggered = match rule.comparison {
                AlertComparison::GreaterThan => value > rule.threshold,
                AlertComparison::LessThan => value < rule.threshold,
                AlertComparison::Equal => (value - rule.threshold).abs() < 0.001,
                AlertComparison::GreaterThanOrEqual => value >= rule.threshold,
                AlertComparison::LessThanOrEqual => value <= rule.threshold,
            };
            
            if triggered {
                self.trigger_alert(rule.clone(), collection.to_string(), value).await;
            }
        }
    }

    /// Trigger an alert
    async fn trigger_alert(&self, rule: AlertRule, collection: String, current_value: f64) {
        let alert_id = Uuid::new_v4();
        let message = format!(
            "Alert '{}' triggered for collection '{}': {} {} {} (current: {})",
            rule.rule_name, collection, rule.metric_name, 
            self.comparison_symbol(&rule.comparison), rule.threshold, current_value
        );
        
        let alert = ActiveAlert {
            alert_id,
            rule,
            collection_name: collection,
            triggered_at: Utc::now(),
            current_value,
            message: message.clone(),
            resolved_at: None,
        };
        
        self.active_alerts.write().await.insert(alert_id, alert);
        
        println!("🚨 ALERT TRIGGERED: {}", message);
    }

    fn comparison_symbol(&self, comparison: &AlertComparison) -> &'static str {
        match comparison {
            AlertComparison::GreaterThan => ">",
            AlertComparison::LessThan => "<",
            AlertComparison::Equal => "==",
            AlertComparison::GreaterThanOrEqual => ">=",
            AlertComparison::LessThanOrEqual => "<=",
        }
    }

    /// Update collection health status
    async fn update_collection_health(&self, collection: &str) {
        let performance_window = self.time_series.read().await.calculate_performance_window(collection, 15);
        
        let (status, score, recommendations) = if let Some(window) = performance_window {
            self.assess_health(&window)
        } else {
            (HealthStatus::Unknown, 0.0, vec!["Insufficient data for health assessment".to_string()])
        };
        
        let active_alerts: Vec<ActiveAlert> = self.active_alerts.read().await
            .values()
            .filter(|alert| alert.collection_name == collection && alert.resolved_at.is_none())
            .cloned()
            .collect();
        
        let health = CollectionHealth {
            collection_name: collection.to_string(),
            status,
            last_check: Utc::now(),
            active_alerts,
            performance_score: score,
            recommendations,
        };
        
        self.collection_health.write().await.insert(collection.to_string(), health);
    }

    /// Assess collection health based on performance metrics
    fn assess_health(&self, window: &PerformanceWindow) -> (HealthStatus, f64, Vec<String>) {
        let mut score: f64 = 1.0;
        let mut recommendations = Vec::new();
        
        // Latency assessment
        if window.avg_query_latency_ms > 100.0 {
            score -= 0.3;
            recommendations.push("High average latency detected. Consider optimizing index configuration.".to_string());
        }
        
        if window.p99_query_latency_ms > 500.0 {
            score -= 0.2;
            recommendations.push("High P99 latency indicates occasional slow queries.".to_string());
        }
        
        // Error rate assessment
        if window.error_rate_percent > 5.0 {
            score -= 0.4;
            recommendations.push("High error rate detected. Check index integrity.".to_string());
        }
        
        // Resource usage assessment
        if window.cpu_usage_percent > 80.0 {
            score -= 0.2;
            recommendations.push("High CPU usage. Consider scaling or optimization.".to_string());
        }
        
        if window.memory_usage_mb > 1000.0 {
            score -= 0.1;
            recommendations.push("High memory usage detected.".to_string());
        }
        
        // Cache performance
        if window.cache_hit_rate_percent < 80.0 {
            score -= 0.2;
            recommendations.push("Low cache hit rate. Consider increasing cache size.".to_string());
        }
        
        // Determine status
        let status = if score >= 0.8 {
            HealthStatus::Healthy
        } else if score >= 0.6 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };
        
        if recommendations.is_empty() {
            recommendations.push("Collection performance is optimal.".to_string());
        }
        
        (status, score.max(0.0), recommendations)
    }

    /// Get performance window for a collection
    pub async fn get_performance_window(&self, collection: &str, window_minutes: u64) -> Option<PerformanceWindow> {
        self.time_series.read().await.calculate_performance_window(collection, window_minutes)
    }

    /// Get collection health
    pub async fn get_collection_health(&self, collection: &str) -> Option<CollectionHealth> {
        self.collection_health.read().await.get(collection).cloned()
    }

    /// Get all collection health statuses
    pub async fn get_all_health_statuses(&self) -> HashMap<String, CollectionHealth> {
        self.collection_health.read().await.clone()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<ActiveAlert> {
        self.active_alerts.read().await.values().cloned().collect()
    }

    /// Get metric history
    pub async fn get_metric_history(&self, collection: &str, metric: &str, hours: u64) -> Vec<MetricDataPoint> {
        self.time_series.read().await.get_metric_history(collection, metric, hours)
    }

    /// Simulate real-time metrics collection
    pub async fn simulate_metrics_collection(&self, collection: &str) {
        // Simulate various performance scenarios
        let scenarios = vec![
            ("normal", 15.0, 0.95, 45.0, 256.0, 95.0),      // Normal operation
            ("high_load", 35.0, 0.88, 78.0, 512.0, 85.0),  // High load
            ("degraded", 85.0, 0.75, 95.0, 768.0, 65.0),   // Degraded performance
        ];
        
        for (scenario_name, latency, success_rate, cpu, memory, cache_hit) in scenarios {
            println!("📈 Simulating {} scenario for collection '{}'", scenario_name, collection);
            
            // Record multiple data points for this scenario
            for i in 0..10 {
                let jitter = (i as f64) * 0.1; // Add some variation
                
                self.record_query_performance(collection, latency + jitter, rand::random::<f64>() < success_rate).await;
                self.record_system_metrics(collection, cpu + jitter, memory + (jitter * 10.0), 128.0).await;
                self.record_metric(collection, "cache_hit_rate", cache_hit + jitter, None).await;
                
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }
    }

    /// Resolve an alert
    pub async fn resolve_alert(&self, alert_id: Uuid) -> Result<(), AppError> {
        if let Some(alert) = self.active_alerts.write().await.get_mut(&alert_id) {
            alert.resolved_at = Some(Utc::now());
            println!("✅ Resolved alert: {}", alert_id);
            Ok(())
        } else {
            Err(AppError::NotFound(format!("Alert {} not found", alert_id)))
        }
    }

    /// Get alert rules
    pub async fn get_alert_rules(&self) -> HashMap<Uuid, AlertRule> {
        self.alert_rules.read().await.clone()
    }
}

/// Factory function to create index monitor
pub fn create_index_monitor(retention_hours: u64) -> IndexMonitor {
    IndexMonitor::new(retention_hours)
}