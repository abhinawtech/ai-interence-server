use ai_interence_server::vector::{IndexProfile, create_index_optimizer};
use tokio;

#[tokio::test]
async fn test_index_profile_config_params() {
    // Test that different profiles generate correct configuration parameters
    let high_accuracy = IndexProfile::HighAccuracy;
    let balanced = IndexProfile::Balanced;
    let fast_query = IndexProfile::FastQuery;

    let high_accuracy_params = high_accuracy.get_config_params();
    let balanced_params = balanced.get_config_params();
    let fast_query_params = fast_query.get_config_params();

    // Verify HighAccuracy profile
    assert_eq!(high_accuracy_params.get("ef_construct"), Some(&200));
    assert_eq!(high_accuracy_params.get("m"), Some(&48));
    assert_eq!(high_accuracy_params.get("ef"), Some(&128));

    // Verify Balanced profile
    assert_eq!(balanced_params.get("ef_construct"), Some(&100));
    assert_eq!(balanced_params.get("m"), Some(&16));
    assert_eq!(balanced_params.get("ef"), Some(&64));

    // Verify FastQuery profile
    assert_eq!(fast_query_params.get("ef_construct"), Some(&64));
    assert_eq!(fast_query_params.get("m"), Some(&8));
    assert_eq!(fast_query_params.get("ef"), Some(&32));

    println!("✅ Index profile configurations are correct");
}

#[tokio::test]
async fn test_index_optimizer_collection_registration() {
    let optimizer = create_index_optimizer();
    
    // Test collection registration
    let result = optimizer.register_collection(
        "test_collection",
        IndexProfile::Balanced,
        true // auto_optimize enabled
    ).await;
    
    assert!(result.is_ok(), "Collection registration should succeed");
    
    // Verify collection is registered
    let configs = optimizer.get_all_configurations().await;
    assert!(configs.contains_key("test_collection"));
    
    let config = configs.get("test_collection").unwrap();
    assert_eq!(config.collection_name, "test_collection");
    assert!(matches!(config.current_profile, IndexProfile::Balanced));
    assert!(config.auto_optimize);
    assert_eq!(config.optimization_threshold, 0.15);
    
    println!("✅ Collection registration works correctly");
}

#[tokio::test]
async fn test_performance_analysis() {
    let optimizer = create_index_optimizer();
    
    // Register collections with different profiles
    optimizer.register_collection("high_accuracy_collection", IndexProfile::HighAccuracy, false).await.unwrap();
    optimizer.register_collection("balanced_collection", IndexProfile::Balanced, false).await.unwrap();
    optimizer.register_collection("fast_query_collection", IndexProfile::FastQuery, false).await.unwrap();
    
    // Analyze performance for each profile
    let high_acc_metrics = optimizer.analyze_collection_performance("high_accuracy_collection").await.unwrap();
    let balanced_metrics = optimizer.analyze_collection_performance("balanced_collection").await.unwrap();
    let fast_metrics = optimizer.analyze_collection_performance("fast_query_collection").await.unwrap();
    
    // Verify performance characteristics
    // HighAccuracy should have higher latency but better accuracy
    assert!(high_acc_metrics.avg_query_latency_ms > balanced_metrics.avg_query_latency_ms);
    assert!(high_acc_metrics.accuracy_score > balanced_metrics.accuracy_score);
    
    // FastQuery should have lowest latency but lower accuracy
    assert!(fast_metrics.avg_query_latency_ms < balanced_metrics.avg_query_latency_ms);
    assert!(fast_metrics.accuracy_score < balanced_metrics.accuracy_score);
    
    // Verify QPS follows inverse relationship with latency
    assert!(fast_metrics.queries_per_second > balanced_metrics.queries_per_second);
    assert!(balanced_metrics.queries_per_second > high_acc_metrics.queries_per_second);
    
    println!("✅ Performance analysis shows correct profile characteristics");
}

#[tokio::test]
async fn test_optimization_recommendations() {
    let optimizer = create_index_optimizer();
    
    // Test different profiles get appropriate recommendations
    optimizer.register_collection("high_latency_collection", IndexProfile::HighAccuracy, false).await.unwrap();
    optimizer.register_collection("low_accuracy_collection", IndexProfile::FastQuery, false).await.unwrap();
    optimizer.register_collection("optimal_collection", IndexProfile::Balanced, false).await.unwrap();
    
    let high_latency_recs = optimizer.get_optimization_recommendations("high_latency_collection").await.unwrap();
    let low_accuracy_recs = optimizer.get_optimization_recommendations("low_accuracy_collection").await.unwrap();
    let optimal_recs = optimizer.get_optimization_recommendations("optimal_collection").await.unwrap();
    
    // High latency collection should get FastQuery recommendation
    assert!(high_latency_recs.iter().any(|r| r.contains("FastQuery")));
    
    // Low accuracy collection should get HighAccuracy recommendation
    assert!(low_accuracy_recs.iter().any(|r| r.contains("HighAccuracy")));
    
    // Optimal collection should be told it's optimal
    assert!(optimal_recs.iter().any(|r| r.contains("optimal")));
    
    println!("✅ Optimization recommendations are contextually appropriate");
}

#[tokio::test]
async fn test_auto_optimization() {
    let optimizer = create_index_optimizer();
    
    // Register collection with auto-optimization enabled
    optimizer.register_collection("auto_optimize_test", IndexProfile::HighAccuracy, true).await.unwrap();
    
    // HighAccuracy now has 35ms latency and 28.6 QPS, which should trigger optimization
    let optimized = optimizer.auto_optimize_collection("auto_optimize_test").await.unwrap();
    
    // Should have optimized to FastQuery due to high latency (35ms > 30ms)
    assert!(optimized, "Auto-optimization should have been triggered due to high latency");
    
    // Verify profile was changed to FastQuery
    let config = optimizer.get_collection_config("auto_optimize_test").await.unwrap();
    assert!(matches!(config.current_profile, IndexProfile::FastQuery));
    
    println!("✅ Auto-optimization correctly changes profiles based on performance");
}

#[tokio::test]
async fn test_benchmark_configurations() {
    let optimizer = create_index_optimizer();
    
    // Benchmark different configurations
    let benchmark_results = optimizer.benchmark_configurations("benchmark_test", 100).await.unwrap();
    
    // Should have results for all three profiles
    assert!(benchmark_results.contains_key("high_accuracy"));
    assert!(benchmark_results.contains_key("balanced"));
    assert!(benchmark_results.contains_key("fast_query"));
    
    let high_acc = benchmark_results.get("high_accuracy").unwrap();
    let balanced = benchmark_results.get("balanced").unwrap();
    let fast_query = benchmark_results.get("fast_query").unwrap();
    
    // Verify expected performance relationships
    assert!(fast_query.avg_query_latency_ms < balanced.avg_query_latency_ms);
    assert!(balanced.avg_query_latency_ms < high_acc.avg_query_latency_ms);
    
    assert!(high_acc.accuracy_score > balanced.accuracy_score);
    assert!(balanced.accuracy_score > fast_query.accuracy_score);
    
    println!("✅ Benchmark provides comprehensive performance comparison");
}

#[tokio::test]
async fn test_profile_updates() {
    let optimizer = create_index_optimizer();
    
    // Register with initial profile
    optimizer.register_collection("update_test", IndexProfile::FastQuery, false).await.unwrap();
    
    // Update to different profile
    let result = optimizer.update_collection_profile("update_test", IndexProfile::HighAccuracy).await;
    assert!(result.is_ok(), "Profile update should succeed");
    
    // Verify profile was updated
    let config = optimizer.get_collection_config("update_test").await.unwrap();
    assert!(matches!(config.current_profile, IndexProfile::HighAccuracy));
    assert!(config.last_optimization.is_some());
    
    // Test updating non-existent collection
    let result = optimizer.update_collection_profile("non_existent", IndexProfile::Balanced).await;
    assert!(result.is_err(), "Updating non-existent collection should fail");
    
    println!("✅ Profile updates work correctly with proper error handling");
}

#[tokio::test]
async fn test_custom_profile() {
    let custom_profile = IndexProfile::Custom {
        ef_construct: 150,
        m: 24,
        ef: 96,
        max_indexing_threads: 8,
    };
    
    let params = custom_profile.get_config_params();
    
    assert_eq!(params.get("ef_construct"), Some(&150));
    assert_eq!(params.get("m"), Some(&24));
    assert_eq!(params.get("ef"), Some(&96));
    assert_eq!(params.get("max_indexing_threads"), Some(&8));
    
    // Test with optimizer
    let optimizer = create_index_optimizer();
    optimizer.register_collection("custom_test", custom_profile, false).await.unwrap();
    
    let config = optimizer.get_collection_config("custom_test").await.unwrap();
    assert!(matches!(config.current_profile, IndexProfile::Custom { .. }));
    
    println!("✅ Custom profiles work correctly with specified parameters");
}