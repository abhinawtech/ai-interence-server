use ai_interence_server::vector::create_index_monitor;
use tokio;

#[tokio::test]
async fn test_basic_monitoring() {
    let monitor = create_index_monitor(24);
    
    // Record a simple metric
    monitor.record_metric("test", "cpu", 50.0, None).await;
    
    // Get history
    let history = monitor.get_metric_history("test", "cpu", 1).await;
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].value, 50.0);
    
    println!("✅ Basic monitoring works");
}