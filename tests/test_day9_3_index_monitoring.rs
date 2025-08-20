use ai_interence_server::vector::{create_index_monitor, AlertSeverity, AlertComparison, AlertRule, HealthStatus};
use std::collections::HashMap;
use uuid::Uuid;
use tokio;

#[tokio::test]
async fn test_index_monitor_creation() {
    let monitor = create_index_monitor(24); // 24 hour retention
    
    // Record some initial metrics
    monitor.record_metric("test_collection", "query_latency_ms", 15.0, None).await;
    monitor.record_metric("test_collection", "cpu_usage_percent", 45.0, None).await;
    
    // Get metric history
    let history = monitor.get_metric_history("test_collection", "query_latency_ms", 1).await;
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].value, 15.0);
    
    println!("✅ Index monitor created and basic metrics recording works");
}

#[tokio::test]
async fn test_performance_window_calculation() {
    let monitor = create_index_monitor(24);
    
    // Record multiple performance metrics
    let latencies = vec![10.0, 15.0, 12.0, 18.0, 25.0, 8.0, 14.0, 20.0];
    for latency in latencies {
        monitor.record_query_performance("perf_test", latency, true).await;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    
    // Record system metrics
    monitor.record_system_metrics("perf_test", 55.0, 512.0, 256.0).await;
    
    // Get performance window
    let window = monitor.get_performance_window("perf_test", 15).await;
    assert!(window.is_some());
    
    let window = window.unwrap();
    assert!(window.avg_query_latency_ms > 0.0);
    assert!(window.p95_query_latency_ms >= window.avg_query_latency_ms);
    assert!(window.p99_query_latency_ms >= window.p95_query_latency_ms);
    assert_eq!(window.index_size_mb, 256.0);
    
    println!("✅ Performance window calculation works: avg={:.1}ms, p95={:.1}ms, p99={:.1}ms", 
            window.avg_query_latency_ms, window.p95_query_latency_ms, window.p99_query_latency_ms);
}

#[tokio::test]
async fn test_alert_system() {
    let monitor = create_index_monitor(24);
    
    // Create alert rule for high latency
    let rule = AlertRule {
        rule_id: Uuid::new_v4(),
        rule_name: "High Latency Alert".to_string(),
        metric_name: "query_latency_ms".to_string(),
        threshold: 50.0,
        comparison: AlertComparison::GreaterThan,
        severity: AlertSeverity::Warning,
        duration_minutes: 5,
        enabled: true,
        collection_pattern: Some("alert_test".to_string()),
    };
    
    monitor.create_alert_rule(rule).await.unwrap();
    
    // Record metric that should trigger alert
    monitor.record_metric("alert_test_collection", "query_latency_ms", 75.0, None).await;
    
    // Give time for alert processing
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Check active alerts
    let active_alerts = monitor.get_active_alerts().await;
    assert!(active_alerts.len() > 0, "Alert should have been triggered");
    
    let alert = &active_alerts[0];
    assert_eq!(alert.current_value, 75.0);
    assert!(alert.message.contains("High Latency Alert"));
    assert!(alert.resolved_at.is_none());
    
    // Resolve the alert
    monitor.resolve_alert(alert.alert_id).await.unwrap();
    
    let resolved_alerts = monitor.get_active_alerts().await;
    let resolved_alert = resolved_alerts.iter().find(|a| a.alert_id == alert.alert_id).unwrap();
    assert!(resolved_alert.resolved_at.is_some());
    
    println!("✅ Alert system works: triggered alert for 75ms > 50ms threshold");
}

#[tokio::test]
async fn test_collection_health_assessment() {
    let monitor = create_index_monitor(24);
    
    // Record metrics for healthy collection
    monitor.record_query_performance("healthy_collection", 12.0, true).await;
    monitor.record_system_metrics("healthy_collection", 35.0, 256.0, 128.0).await;
    monitor.record_metric("healthy_collection", "cache_hit_rate", 95.0, None).await;
    
    let health = monitor.get_collection_health("healthy_collection").await;
    assert!(health.is_some());
    
    let health = health.unwrap();
    assert_eq!(health.collection_name, "healthy_collection");
    assert!(matches!(health.status, HealthStatus::Healthy));
    assert!(health.performance_score > 0.8);
    assert!(health.recommendations.iter().any(|r| r.contains("optimal")));
    
    // Record metrics for degraded collection
    monitor.record_query_performance("degraded_collection", 150.0, false).await; // High latency + error
    monitor.record_system_metrics("degraded_collection", 95.0, 1200.0, 512.0).await; // High CPU + memory
    monitor.record_metric("degraded_collection", "cache_hit_rate", 65.0, None).await; // Low cache hit
    monitor.record_metric("degraded_collection", "error_rate", 15.0, None).await; // High error rate
    
    let degraded_health = monitor.get_collection_health("degraded_collection").await.unwrap();
    assert!(matches!(degraded_health.status, HealthStatus::Critical));
    assert!(degraded_health.performance_score < 0.6);
    assert!(degraded_health.recommendations.len() > 3); // Should have multiple recommendations
    
    println!("✅ Health assessment works: healthy={:.2} score, degraded={:.2} score", 
            health.performance_score, degraded_health.performance_score);
}

#[tokio::test]
async fn test_metric_history_and_retention() {
    let monitor = create_index_monitor(1); // 1 hour retention for testing
    
    let collection = "history_test";
    
    // Record metrics over time
    for i in 0..5 {
        let value = 10.0 + (i as f64);
        let mut tags = HashMap::new();
        tags.insert("test_iteration".to_string(), i.to_string());
        
        monitor.record_metric(collection, "test_metric", value, Some(tags)).await;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    
    // Get metric history
    let history = monitor.get_metric_history(collection, "test_metric", 1).await;
    assert_eq!(history.len(), 5);
    
    // Verify values are in order
    for (i, point) in history.iter().enumerate() {
        assert_eq!(point.value, 10.0 + (i as f64));
        assert!(point.tags.contains_key("test_iteration"));
    }
    
    // Test different time windows
    let short_history = monitor.get_metric_history(collection, "test_metric", 0).await;
    assert!(short_history.len() <= history.len());
    
    println!("✅ Metric history tracking works with {} data points", history.len());
}

#[tokio::test]
async fn test_multiple_alert_rules() {
    let monitor = create_index_monitor(24);
    
    // Create multiple alert rules
    let rules = vec![
        AlertRule {
            rule_id: Uuid::new_v4(),
            rule_name: "CPU Alert".to_string(),
            metric_name: "cpu_usage_percent".to_string(),
            threshold: 80.0,
            comparison: AlertComparison::GreaterThan,
            severity: AlertSeverity::Critical,
            duration_minutes: 10,
            enabled: true,
            collection_pattern: None,
        },
        AlertRule {
            rule_id: Uuid::new_v4(),
            rule_name: "Memory Alert".to_string(),
            metric_name: "memory_usage_mb".to_string(),
            threshold: 1000.0,
            comparison: AlertComparison::GreaterThanOrEqual,
            severity: AlertSeverity::Warning,
            duration_minutes: 15,
            enabled: true,
            collection_pattern: None,
        },
        AlertRule {
            rule_id: Uuid::new_v4(),
            rule_name: "Low Cache Alert".to_string(),
            metric_name: "cache_hit_rate".to_string(),
            threshold: 70.0,
            comparison: AlertComparison::LessThan,
            severity: AlertSeverity::Info,
            duration_minutes: 30,
            enabled: true,
            collection_pattern: None,
        },
    ];
    
    for rule in rules {
        monitor.create_alert_rule(rule).await.unwrap();
    }
    
    // Get all alert rules
    let all_rules = monitor.get_alert_rules().await;
    assert_eq!(all_rules.len(), 3);
    
    // Trigger multiple alerts
    monitor.record_metric("multi_alert_test", "cpu_usage_percent", 85.0, None).await;
    monitor.record_metric("multi_alert_test", "memory_usage_mb", 1024.0, None).await;
    monitor.record_metric("multi_alert_test", "cache_hit_rate", 65.0, None).await;
    
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    let active_alerts = monitor.get_active_alerts().await;
    assert!(active_alerts.len() >= 3, "Should have triggered 3 alerts, got {}", active_alerts.len());
    
    // Verify different severity levels
    let severities: Vec<_> = active_alerts.iter().map(|a| &a.rule.severity).collect();
    assert!(severities.contains(&&AlertSeverity::Critical));
    assert!(severities.contains(&&AlertSeverity::Warning));
    assert!(severities.contains(&&AlertSeverity::Info));
    
    println!("✅ Multiple alert rules work: {} rules created, {} alerts triggered", 
            all_rules.len(), active_alerts.len());
}

#[tokio::test]
async fn test_alert_comparison_operators() {
    let monitor = create_index_monitor(24);
    
    // Test all comparison operators
    let comparisons = vec![
        (AlertComparison::GreaterThan, 50.0, 60.0, true),      // 60 > 50
        (AlertComparison::LessThan, 50.0, 40.0, true),         // 40 < 50
        (AlertComparison::Equal, 50.0, 50.0, true),            // 50 == 50
        (AlertComparison::GreaterThanOrEqual, 50.0, 50.0, true), // 50 >= 50
        (AlertComparison::LessThanOrEqual, 50.0, 50.0, true),   // 50 <= 50
        (AlertComparison::GreaterThan, 50.0, 40.0, false),     // 40 > 50 (false)
    ];
    
    for (i, (comparison, threshold, test_value, should_trigger)) in comparisons.iter().enumerate() {
        let rule = AlertRule {
            rule_id: Uuid::new_v4(),
            rule_name: format!("Comparison Test {}", i),
            metric_name: "comparison_test".to_string(),
            threshold: *threshold,
            comparison: comparison.clone(),
            severity: AlertSeverity::Info,
            duration_minutes: 1,
            enabled: true,
            collection_pattern: None,
        };
        
        monitor.create_alert_rule(rule).await.unwrap();
        
        let collection = format!("comparison_test_{}", i);
        monitor.record_metric(&collection, "comparison_test", *test_value, None).await;
        
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        let alerts = monitor.get_active_alerts().await;
        let collection_alerts: Vec<_> = alerts.iter()
            .filter(|a| a.collection_name == collection)
            .collect();
        
        if *should_trigger {
            assert!(!collection_alerts.is_empty(), 
                   "Alert should have triggered for {:?} {} {} with value {}", 
                   comparison, threshold, if *should_trigger { "=>" } else { "!>" }, test_value);
        } else {
            assert!(collection_alerts.is_empty(), 
                   "Alert should NOT have triggered for {:?} {} {} with value {}", 
                   comparison, threshold, if *should_trigger { "=>" } else { "!>" }, test_value);
        }
    }
    
    println!("✅ Alert comparison operators work correctly");
}

#[tokio::test]
async fn test_metrics_simulation() {
    let monitor = create_index_monitor(24);
    
    // Run metrics simulation
    monitor.simulate_metrics_collection("simulation_test").await;
    
    // Check that metrics were recorded
    let latency_history = monitor.get_metric_history("simulation_test", "query_latency_ms", 1).await;
    let cpu_history = monitor.get_metric_history("simulation_test", "cpu_usage_percent", 1).await;
    let cache_history = monitor.get_metric_history("simulation_test", "cache_hit_rate", 1).await;
    
    assert!(latency_history.len() >= 10, "Should have multiple latency data points");
    assert!(cpu_history.len() >= 10, "Should have multiple CPU data points");
    assert!(cache_history.len() >= 10, "Should have multiple cache data points");
    
    // Verify we have different scenarios (normal, high_load, degraded)
    let latencies: Vec<f64> = latency_history.iter().map(|p| p.value).collect();
    let min_latency = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_latency = latencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    assert!(max_latency > min_latency + 10.0, "Should have varied latency values");
    
    // Check collection health after simulation
    let health = monitor.get_collection_health("simulation_test").await;
    assert!(health.is_some());
    
    let health = health.unwrap();
    assert!(!health.recommendations.is_empty());
    
    println!("✅ Metrics simulation works: {} latency points, range {:.1}-{:.1}ms, health: {:?}", 
            latency_history.len(), min_latency, max_latency, health.status);
}

#[tokio::test]
async fn test_collection_pattern_matching() {
    let monitor = create_index_monitor(24);
    
    // Create alert rule with collection pattern
    let rule = AlertRule {
        rule_id: Uuid::new_v4(),
        rule_name: "Pattern Test Alert".to_string(),
        metric_name: "test_metric".to_string(),
        threshold: 100.0,
        comparison: AlertComparison::GreaterThan,
        severity: AlertSeverity::Warning,
        duration_minutes: 5,
        enabled: true,
        collection_pattern: Some("production".to_string()), // Only match collections with "production"
    };
    
    monitor.create_alert_rule(rule).await.unwrap();
    
    // Record metrics for collections with and without pattern
    monitor.record_metric("production_db", "test_metric", 150.0, None).await;
    monitor.record_metric("test_db", "test_metric", 150.0, None).await;
    
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    let alerts = monitor.get_active_alerts().await;
    
    // Should only trigger for production_db
    let production_alerts: Vec<_> = alerts.iter()
        .filter(|a| a.collection_name == "production_db")
        .collect();
    let test_alerts: Vec<_> = alerts.iter()
        .filter(|a| a.collection_name == "test_db")
        .collect();
    
    assert!(!production_alerts.is_empty(), "Should trigger alert for production_db");
    assert!(test_alerts.is_empty(), "Should NOT trigger alert for test_db");
    
    println!("✅ Collection pattern matching works: production collection triggered alert, test collection did not");
}

#[tokio::test]
async fn test_all_health_statuses() {
    let monitor = create_index_monitor(24);
    
    // Create collections with different health profiles
    let collections = vec![
        ("excellent_collection", 8.0, 98.0, 30.0, 200.0),    // Excellent
        ("good_collection", 18.0, 95.0, 55.0, 400.0),        // Good  
        ("warning_collection", 45.0, 85.0, 75.0, 800.0),     // Warning
        ("critical_collection", 120.0, 60.0, 95.0, 1500.0),  // Critical
    ];
    
    for (name, latency, cache_hit, cpu, memory) in collections {
        monitor.record_query_performance(name, latency, true).await;
        monitor.record_system_metrics(name, cpu, memory, 256.0).await;
        monitor.record_metric(name, "cache_hit_rate", cache_hit, None).await;
    }
    
    let all_health = monitor.get_all_health_statuses().await;
    assert_eq!(all_health.len(), 4);
    
    // Verify different health statuses
    let excellent = all_health.get("excellent_collection").unwrap();
    let critical = all_health.get("critical_collection").unwrap();
    
    assert!(matches!(excellent.status, HealthStatus::Healthy));
    assert!(matches!(critical.status, HealthStatus::Critical));
    assert!(excellent.performance_score > critical.performance_score);
    
    println!("✅ All health statuses work: {} collections monitored", all_health.len());
}