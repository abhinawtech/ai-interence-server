// ARCHITECTURE: Security Module - Production-Grade API Protection System
//
// DESIGN PHILOSOPHY:
// This module implements enterprise-grade security features designed for:
// 1. RATE LIMITING: Prevent API abuse and DDoS attacks with token bucket algorithm
// 2. AUTHENTICATION: Secure API access with API key validation
// 3. AUTHORIZATION: Role-based access control for different API endpoints
// 4. AUDIT LOGGING: Comprehensive security event logging for compliance
// 5. THREAT PROTECTION: Real-time threat detection and response
//
// SECURITY FEATURES:
// - Token bucket rate limiting with per-client limits
// - API key authentication with configurable expiration
// - IP-based rate limiting with geographic restrictions
// - Request signature validation for critical operations
// - Comprehensive audit trails for security monitoring
//
// PRODUCTION REQUIREMENTS MET:
// ✅ DDoS protection with configurable rate limits
// ✅ API key authentication for unauthorized access prevention
// ✅ Per-client rate limiting with burst allowance
// ✅ Security audit logging for compliance
// ✅ Configurable security policies for different environments

pub mod rate_limiter;
pub mod auth;
pub mod middleware;

pub use rate_limiter::{RateLimiter, RateLimiterConfig, RateLimitResult};
pub use auth::{AuthService, ApiKey, AuthResult};
pub use middleware::{SecurityMiddleware, SecurityConfig};