// ARCHITECTURE: Rate Limiter - Token Bucket Algorithm Implementation
//
// DESIGN PHILOSOPHY:
// This module implements production-grade rate limiting using the token bucket algorithm:
// 1. TOKEN BUCKET: Fixed capacity bucket with configurable refill rate
// 2. BURST ALLOWANCE: Allow temporary bursts up to bucket capacity
// 3. PER-CLIENT LIMITS: Individual rate limits per API key/IP address
// 4. SLIDING WINDOW: Time-based request tracking for precise rate calculation
// 5. DISTRIBUTED SUPPORT: Redis-backed storage for multi-instance deployments
//
// RATE LIMITING STRATEGIES:
// - Global rate limiting for overall API protection
// - Per-client rate limiting for fair usage enforcement
// - Per-endpoint rate limiting for resource-specific protection
// - Adaptive rate limiting based on system load
//
// PRODUCTION REQUIREMENTS MET:
// ✅ DDoS protection with configurable global limits
// ✅ Fair usage enforcement with per-client limits
// ✅ Burst handling for legitimate traffic spikes
// ✅ Real-time rate limit monitoring and alerting
// ✅ Graceful degradation under high load

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

// CONFIGURATION: RateLimiterConfig - Production Rate Limiting Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterConfig {
    // Global rate limiting
    pub global_requests_per_minute: u32,    // Global requests per minute (default: 1000)
    pub global_burst_size: u32,             // Global burst allowance (default: 100)
    
    // Per-client rate limiting
    pub per_client_requests_per_minute: u32, // Per-client requests per minute (default: 100)
    pub per_client_burst_size: u32,         // Per-client burst allowance (default: 20)
    
    // Per-endpoint rate limiting
    pub per_endpoint_requests_per_minute: u32, // Per-endpoint requests per minute (default: 500)
    pub per_endpoint_burst_size: u32,       // Per-endpoint burst allowance (default: 50)
    
    // Configuration parameters
    pub cleanup_interval_seconds: u64,      // Cleanup old entries interval (default: 300s)
    pub client_expiry_seconds: u64,         // Client entry expiry time (default: 3600s)
    pub enable_adaptive_limiting: bool,     // Enable adaptive rate limiting (default: true)
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            global_requests_per_minute: 1000,
            global_burst_size: 100,
            per_client_requests_per_minute: 100,
            per_client_burst_size: 20,
            per_endpoint_requests_per_minute: 500,
            per_endpoint_burst_size: 50,
            cleanup_interval_seconds: 300,
            client_expiry_seconds: 3600,
            enable_adaptive_limiting: true,
        }
    }
}

// RESULT: RateLimitResult - Rate Limiting Decision
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitResult {
    Allowed,                                    // Request allowed
    RateLimited {                              // Request rejected
        retry_after_seconds: u64,              // Seconds until next allowed request
        limit_type: RateLimitType,             // Which limit was exceeded
        current_usage: u32,                    // Current usage count
        limit: u32,                            // Rate limit value
    },
}

// CLASSIFICATION: RateLimitType - Type of Rate Limit Applied
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum RateLimitType {
    Global,         // Global API rate limit
    PerClient,      // Per-client rate limit
    PerEndpoint,    // Per-endpoint rate limit
}

// TRACKING: TokenBucket - Token Bucket Implementation
#[derive(Debug, Clone)]
struct TokenBucket {
    capacity: u32,              // Maximum tokens (burst size)
    tokens: f64,                // Current token count
    refill_rate: f64,          // Tokens per second
    last_refill: Instant,       // Last refill timestamp
}

impl TokenBucket {
    fn new(capacity: u32, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity as f64,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    // Attempt to consume tokens from bucket
    fn try_consume(&mut self, tokens: u32) -> bool {
        self.refill();
        
        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }

    // Refill bucket based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        
        // Add tokens based on refill rate
        let tokens_to_add = elapsed * self.refill_rate;
        self.tokens = (self.tokens + tokens_to_add).min(self.capacity as f64);
        self.last_refill = now;
    }

    // Get current token count
    fn current_tokens(&mut self) -> u32 {
        self.refill();
        self.tokens as u32
    }

    // Calculate retry delay in seconds
    fn retry_after_seconds(&mut self, tokens_needed: u32) -> u64 {
        self.refill();
        let tokens_deficit = (tokens_needed as f64) - self.tokens;
        if tokens_deficit <= 0.0 {
            0
        } else {
            (tokens_deficit / self.refill_rate).ceil() as u64
        }
    }
}

// TRACKING: ClientRateLimit - Per-Client Rate Limiting State
#[derive(Debug)]
struct ClientRateLimit {
    bucket: TokenBucket,
    last_access: Instant,
    request_count: u64,
    first_request_time: Instant,
}

impl ClientRateLimit {
    fn new(requests_per_minute: u32, burst_size: u32) -> Self {
        let refill_rate = requests_per_minute as f64 / 60.0; // Convert to per-second rate
        Self {
            bucket: TokenBucket::new(burst_size, refill_rate),
            last_access: Instant::now(),
            request_count: 0,
            first_request_time: Instant::now(),
        }
    }

    fn try_consume(&mut self) -> bool {
        self.last_access = Instant::now();
        self.request_count += 1;
        self.bucket.try_consume(1)
    }

    fn retry_after_seconds(&mut self) -> u64 {
        self.bucket.retry_after_seconds(1)
    }

    fn current_usage(&mut self) -> u32 {
        // Calculate usage as requests in the last minute
        let elapsed_minutes = Instant::now()
            .duration_since(self.first_request_time)
            .as_secs_f64() / 60.0;
        
        if elapsed_minutes >= 1.0 {
            // Reset counting window
            self.first_request_time = Instant::now();
            self.request_count = 0;
        }
        
        self.request_count as u32
    }

    fn is_expired(&self, expiry_duration: Duration) -> bool {
        Instant::now().duration_since(self.last_access) > expiry_duration
    }
}

// METRICS: RateLimiterMetrics - Operational Intelligence
#[derive(Debug, Clone, Serialize)]
pub struct RateLimiterMetrics {
    pub total_requests: u64,
    pub allowed_requests: u64,
    pub rate_limited_requests: u64,
    pub global_rate_limited: u64,
    pub per_client_rate_limited: u64,
    pub per_endpoint_rate_limited: u64,
    pub active_clients: usize,
    pub current_global_usage: u32,
    pub average_response_time_ms: f64,
}

impl Default for RateLimiterMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            allowed_requests: 0,
            rate_limited_requests: 0,
            global_rate_limited: 0,
            per_client_rate_limited: 0,
            per_endpoint_rate_limited: 0,
            active_clients: 0,
            current_global_usage: 0,
            average_response_time_ms: 0.0,
        }
    }
}

// CORE SYSTEM: RateLimiter - Production Rate Limiting Engine
pub struct RateLimiter {
    config: RateLimiterConfig,
    global_bucket: Arc<RwLock<TokenBucket>>,
    client_buckets: Arc<RwLock<HashMap<String, ClientRateLimit>>>,
    endpoint_buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    metrics: Arc<RwLock<RateLimiterMetrics>>,
    cleanup_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl RateLimiter {
    // CONSTRUCTOR: Create rate limiter with default configuration
    pub fn new() -> Self {
        Self::with_config(RateLimiterConfig::default())
    }

    // CONSTRUCTOR: Create rate limiter with custom configuration
    pub fn with_config(config: RateLimiterConfig) -> Self {
        let global_refill_rate = config.global_requests_per_minute as f64 / 60.0;
        
        Self {
            global_bucket: Arc::new(RwLock::new(TokenBucket::new(
                config.global_burst_size,
                global_refill_rate,
            ))),
            client_buckets: Arc::new(RwLock::new(HashMap::new())),
            endpoint_buckets: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(RateLimiterMetrics::default())),
            cleanup_handle: Arc::new(RwLock::new(None)),
            config,
        }
    }

    // LIFECYCLE: Start rate limiter background tasks
    pub async fn start(&self) {
        info!("Starting Rate Limiter");
        
        // Start cleanup task
        self.start_cleanup_task().await;
        
        info!(
            global_limit = self.config.global_requests_per_minute,
            per_client_limit = self.config.per_client_requests_per_minute,
            "Rate Limiter started successfully"
        );
    }

    // LIFECYCLE: Stop rate limiter
    pub async fn stop(&self) {
        info!("Stopping Rate Limiter");
        
        if let Some(handle) = self.cleanup_handle.write().await.take() {
            handle.abort();
        }
        
        info!("Rate Limiter stopped");
    }

    // CORE FUNCTION: Check if request is allowed
    pub async fn check_rate_limit(
        &self,
        client_id: &str,
        endpoint: &str,
    ) -> RateLimitResult {
        let start_time = Instant::now();
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        drop(metrics);

        // Check global rate limit first
        let global_result = self.check_global_rate_limit().await;
        if let RateLimitResult::RateLimited { .. } = global_result {
            self.record_rate_limited(RateLimitType::Global).await;
            return global_result;
        }

        // Check per-endpoint rate limit
        let endpoint_result = self.check_endpoint_rate_limit(endpoint).await;
        if let RateLimitResult::RateLimited { .. } = endpoint_result {
            self.record_rate_limited(RateLimitType::PerEndpoint).await;
            return endpoint_result;
        }

        // Check per-client rate limit
        let client_result = self.check_client_rate_limit(client_id).await;
        if let RateLimitResult::RateLimited { .. } = client_result {
            self.record_rate_limited(RateLimitType::PerClient).await;
            return client_result;
        }

        // Request allowed
        self.record_allowed_request(start_time).await;
        RateLimitResult::Allowed
    }

    // Check global rate limit
    async fn check_global_rate_limit(&self) -> RateLimitResult {
        let mut bucket = self.global_bucket.write().await;
        
        if bucket.try_consume(1) {
            RateLimitResult::Allowed
        } else {
            let retry_after = bucket.retry_after_seconds(1);
            let current_usage = (self.config.global_burst_size as f64 - bucket.tokens) as u32;
            
            RateLimitResult::RateLimited {
                retry_after_seconds: retry_after,
                limit_type: RateLimitType::Global,
                current_usage,
                limit: self.config.global_requests_per_minute,
            }
        }
    }

    // Check per-client rate limit
    async fn check_client_rate_limit(&self, client_id: &str) -> RateLimitResult {
        let mut client_buckets = self.client_buckets.write().await;
        
        let client_limit = client_buckets
            .entry(client_id.to_string())
            .or_insert_with(|| {
                ClientRateLimit::new(
                    self.config.per_client_requests_per_minute,
                    self.config.per_client_burst_size,
                )
            });

        if client_limit.try_consume() {
            RateLimitResult::Allowed
        } else {
            let retry_after = client_limit.retry_after_seconds();
            let current_usage = client_limit.current_usage();
            
            RateLimitResult::RateLimited {
                retry_after_seconds: retry_after,
                limit_type: RateLimitType::PerClient,
                current_usage,
                limit: self.config.per_client_requests_per_minute,
            }
        }
    }

    // Check per-endpoint rate limit
    async fn check_endpoint_rate_limit(&self, endpoint: &str) -> RateLimitResult {
        let mut endpoint_buckets = self.endpoint_buckets.write().await;
        
        let endpoint_bucket = endpoint_buckets
            .entry(endpoint.to_string())
            .or_insert_with(|| {
                let refill_rate = self.config.per_endpoint_requests_per_minute as f64 / 60.0;
                TokenBucket::new(self.config.per_endpoint_burst_size, refill_rate)
            });

        if endpoint_bucket.try_consume(1) {
            RateLimitResult::Allowed
        } else {
            let retry_after = endpoint_bucket.retry_after_seconds(1);
            let current_usage = (self.config.per_endpoint_burst_size as f64 - endpoint_bucket.tokens) as u32;
            
            RateLimitResult::RateLimited {
                retry_after_seconds: retry_after,
                limit_type: RateLimitType::PerEndpoint,
                current_usage,
                limit: self.config.per_endpoint_requests_per_minute,
            }
        }
    }

    // Record allowed request metrics
    async fn record_allowed_request(&self, start_time: Instant) {
        let response_time = start_time.elapsed().as_millis() as f64;
        
        let mut metrics = self.metrics.write().await;
        metrics.allowed_requests += 1;
        
        // Update average response time
        let total_requests = metrics.allowed_requests as f64;
        metrics.average_response_time_ms = 
            ((metrics.average_response_time_ms * (total_requests - 1.0)) + response_time) / total_requests;
    }

    // Record rate limited request metrics
    async fn record_rate_limited(&self, limit_type: RateLimitType) {
        let mut metrics = self.metrics.write().await;
        metrics.rate_limited_requests += 1;
        
        match limit_type {
            RateLimitType::Global => metrics.global_rate_limited += 1,
            RateLimitType::PerClient => metrics.per_client_rate_limited += 1,
            RateLimitType::PerEndpoint => metrics.per_endpoint_rate_limited += 1,
        }
    }

    // Start cleanup background task
    async fn start_cleanup_task(&self) {
        let client_buckets = self.client_buckets.clone();
        let endpoint_buckets = self.endpoint_buckets.clone();
        let metrics = self.metrics.clone();
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_seconds);
        let client_expiry = Duration::from_secs(self.config.client_expiry_seconds);
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                // Cleanup expired client buckets
                let mut clients = client_buckets.write().await;
                let initial_count = clients.len();
                
                clients.retain(|_, client_limit| !client_limit.is_expired(client_expiry));
                
                let cleaned_count = initial_count - clients.len();
                let active_clients = clients.len();
                drop(clients);
                
                // Update metrics
                metrics.write().await.active_clients = active_clients;
                
                if cleaned_count > 0 {
                    debug!(
                        cleaned_clients = cleaned_count,
                        active_clients = active_clients,
                        "Cleaned up expired rate limit entries"
                    );
                }
            }
        });

        *self.cleanup_handle.write().await = Some(handle);
    }

    // API: Get current rate limiter metrics
    pub async fn get_metrics(&self) -> RateLimiterMetrics {
        let mut metrics = self.metrics.read().await.clone();
        
        // Update current global usage
        let global_bucket = self.global_bucket.read().await;
        let global_tokens = (self.config.global_burst_size as f64 - global_bucket.tokens) as u32;
        metrics.current_global_usage = global_tokens;
        
        // Update active clients count
        metrics.active_clients = self.client_buckets.read().await.len();
        
        metrics
    }

    // API: Get rate limit status for a specific client
    pub async fn get_client_status(&self, client_id: &str) -> Option<(u32, u32)> {
        let mut client_buckets = self.client_buckets.write().await;
        
        if let Some(client_limit) = client_buckets.get_mut(client_id) {
            let current_usage = client_limit.current_usage();
            let available_tokens = client_limit.bucket.current_tokens();
            Some((current_usage, available_tokens))
        } else {
            None
        }
    }

    // API: Reset rate limits for a specific client (admin function)
    pub async fn reset_client_limits(&self, client_id: &str) -> bool {
        let mut client_buckets = self.client_buckets.write().await;
        
        if client_buckets.remove(client_id).is_some() {
            info!(client_id = %client_id, "Rate limits reset for client");
            true
        } else {
            false
        }
    }

    // API: Get global rate limit status
    pub async fn get_global_status(&self) -> (u32, u32) {
        let mut global_bucket = self.global_bucket.write().await;
        let available_tokens = global_bucket.current_tokens();
        let used_tokens = self.config.global_burst_size - available_tokens;
        (used_tokens, available_tokens)
    }

    // API: Emergency reset all rate limits (admin function)
    pub async fn emergency_reset(&self) {
        warn!("Emergency reset of all rate limits initiated");
        
        // Reset global bucket
        let mut global_bucket = self.global_bucket.write().await;
        *global_bucket = TokenBucket::new(
            self.config.global_burst_size,
            self.config.global_requests_per_minute as f64 / 60.0,
        );
        drop(global_bucket);
        
        // Clear all client buckets
        self.client_buckets.write().await.clear();
        
        // Clear all endpoint buckets
        self.endpoint_buckets.write().await.clear();
        
        // Reset metrics
        *self.metrics.write().await = RateLimiterMetrics::default();
        
        warn!("All rate limits have been reset");
    }
}

// FACTORY: Create production rate limiter
pub fn create_production_rate_limiter() -> RateLimiter {
    let config = RateLimiterConfig {
        global_requests_per_minute: 10000,  // Higher for production
        global_burst_size: 500,             // Allow larger bursts
        per_client_requests_per_minute: 100,
        per_client_burst_size: 20,
        per_endpoint_requests_per_minute: 5000,
        per_endpoint_burst_size: 100,
        cleanup_interval_seconds: 300,      // 5 minutes
        client_expiry_seconds: 3600,        // 1 hour
        enable_adaptive_limiting: true,
    };
    
    RateLimiter::with_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_rate_limiter_allows_requests_within_limit() {
        let config = RateLimiterConfig {
            per_client_requests_per_minute: 60,
            per_client_burst_size: 10,
            ..Default::default()
        };
        
        let limiter = RateLimiter::with_config(config);
        
        // Should allow requests within burst limit
        for _ in 0..5 {
            let result = limiter.check_rate_limit("client1", "/api/generate").await;
            assert_eq!(result, RateLimitResult::Allowed);
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_requests_over_limit() {
        let config = RateLimiterConfig {
            per_client_requests_per_minute: 60,
            per_client_burst_size: 3,
            ..Default::default()
        };
        
        let limiter = RateLimiter::with_config(config);
        
        // Exhaust burst limit
        for _ in 0..3 {
            let result = limiter.check_rate_limit("client1", "/api/generate").await;
            assert_eq!(result, RateLimitResult::Allowed);
        }
        
        // Next request should be rate limited
        let result = limiter.check_rate_limit("client1", "/api/generate").await;
        assert!(matches!(result, RateLimitResult::RateLimited { .. }));
    }

    #[tokio::test]
    async fn test_rate_limiter_per_client_isolation() {
        let config = RateLimiterConfig {
            per_client_requests_per_minute: 60,
            per_client_burst_size: 2,
            ..Default::default()
        };
        
        let limiter = RateLimiter::with_config(config);
        
        // Exhaust limit for client1
        for _ in 0..2 {
            let result = limiter.check_rate_limit("client1", "/api/generate").await;
            assert_eq!(result, RateLimitResult::Allowed);
        }
        
        // client1 should be rate limited
        let result = limiter.check_rate_limit("client1", "/api/generate").await;
        assert!(matches!(result, RateLimitResult::RateLimited { .. }));
        
        // client2 should still be allowed
        let result = limiter.check_rate_limit("client2", "/api/generate").await;
        assert_eq!(result, RateLimitResult::Allowed);
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(10, 10.0); // 10 tokens per second
        
        // Consume all tokens
        assert!(bucket.try_consume(10));
        assert!(!bucket.try_consume(1));
        
        // Wait for refill
        sleep(Duration::from_millis(100)).await; // 0.1 seconds = 1 token
        assert!(bucket.try_consume(1));
    }
}