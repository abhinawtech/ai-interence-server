# Day 9 Vector Database Optimization API Testing Guide

This document provides comprehensive testing examples for the Day 9.1-9.3 index management and optimization features via REST APIs.

## 🚀 Prerequisites

Start the AI inference server:
```bash
cd /Users/abhinawkumar/Code/ai-cloud-native/ai-interence-server
cargo run
```

The server will be available at `http://localhost:3000` (updated port)

## 📊 Day 9.1: Intelligent Index Configuration Optimization

### 1. Collection Registration with Workload Analysis

**Purpose**: Registers a collection and receives AI-driven index configuration recommendations based on query patterns.

```bash
curl -X POST "http://localhost:3000/api/v1/index/collections/register" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "company_documents",
    "vector_size": 384,
    "current_doc_count": 25000,
    "expected_growth_rate": 1000,
    "query_patterns": {
      "high_recall_queries": 65,
      "speed_critical_queries": 25,
      "balanced_queries": 10
    },
    "workload_characteristics": {
      "avg_queries_per_second": 150,
      "peak_queries_per_second": 500,
      "batch_indexing_frequency": "daily"
    }
  }'
```

Expected Response with Analytical Recommendations:
```json
{
  "message": "Collection registered with intelligent optimization",
  "recommended_profile": "HighAccuracy",
  "analysis_summary": {
    "workload_type": "RecallOptimized",
    "performance_prediction": {
      "expected_p99_latency_ms": 45,
      "memory_usage_estimate_gb": 3.2,
      "indexing_throughput_docs_per_sec": 2500
    }
  },
  "config": {
    "hnsw_config": {
      "m": 64,
      "ef_construct": 200,
      "ef_search": 100,
      "max_indexing_threads": 8
    },
    "quantization": {
      "enabled": false,
      "reason": "High recall requirement conflicts with quantization"
    },
    "memory_optimization": {
      "enable_mmap": true,
      "cache_size_mb": 512
    }
  },
  "cost_analysis": {
    "estimated_memory_cost_per_month": "$45",
    "cpu_utilization_percent": 35,
    "storage_efficiency_ratio": 0.85
  }
}
```

### 2. Dynamic Optimization Recommendations

**Purpose**: Continuous analysis of collection performance with actionable recommendations.

```bash
curl -X GET "http://localhost:3000/api/v1/index/collections/company_documents/recommendations"
```

Expected Response:
```json
{
  "collection_name": "company_documents",
  "analysis_timestamp": "2024-12-20T10:45:00Z",
  "current_performance": {
    "query_latency_p50_ms": 12.5,
    "query_latency_p99_ms": 67.2,
    "recall_at_10": 0.94,
    "memory_usage_gb": 2.8
  },
  "recommendations": [
    {
      "type": "PerformanceOptimization",
      "priority": "High",
      "recommendation": "Increase ef_search parameter from 100 to 120",
      "expected_improvement": {
        "recall_increase": 0.03,
        "latency_increase_ms": 5.2,
        "trade_off_analysis": "3% recall improvement for 8% latency cost"
      }
    },
    {
      "type": "CostOptimization", 
      "priority": "Medium",
      "recommendation": "Enable scalar quantization",
      "expected_improvement": {
        "memory_reduction_percent": 40,
        "recall_degradation": 0.015,
        "cost_savings_per_month": "$18"
      }
    }
  ],
  "system_health_score": 87,
  "next_review_recommended": "2024-12-27T10:45:00Z"
}
```

### 3. Profile Application with Impact Analysis

**Purpose**: Apply optimization profiles with predictive impact assessment.

```bash
curl -X POST "http://localhost:3000/api/v1/index/collections/company_documents/apply-profile" \
  -H "Content-Type: application/json" \
  -d '{
    "profile": "FastQuery",
    "force_reindex": false,
    "impact_analysis": true
  }'
```

Expected Response:
```json
{
  "message": "Profile applied successfully",
  "applied_profile": "FastQuery",
  "configuration_changes": {
    "m": {"old": 64, "new": 32},
    "ef_construct": {"old": 200, "new": 100},
    "ef_search": {"old": 100, "new": 64}
  },
  "predicted_impact": {
    "latency_improvement_percent": 35,
    "recall_degradation_percent": 2.1,
    "memory_reduction_gb": 1.1,
    "reindex_required": false
  },
  "rollback_available": true,
  "rollback_token": "rb_550e8400"
}
```

## ⚙️ Day 9.2: Enterprise Background Reindexing

### 1. Intelligent Job Scheduling

**Purpose**: Schedule reindex jobs with resource planning and SLA management.

```bash
curl -X POST "http://localhost:3000/api/v1/reindex/schedule" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "company_documents",
    "priority": "High",
    "scheduling_preferences": {
      "preferred_start_time": "2024-12-20T22:00:00Z",
      "max_delay_hours": 24,
      "allow_business_hours": false
    },
    "resource_requirements": {
      "cpu_cores": 6,
      "memory_gb": 12.0,
      "disk_io_mbps": 800,
      "network_bandwidth_mbps": 100
    },
    "optimization_target": "HighAccuracy",
    "sla_requirements": {
      "max_duration_hours": 4,
      "availability_during_reindex": "ReadOnly"
    }
  }'
```

Expected Response:
```json
{
  "job_id": "job_550e8400-e29b-41d4-a716-446655440000",
  "message": "Reindex job scheduled with resource optimization",
  "scheduling_details": {
    "estimated_start_time": "2024-12-20T22:00:00Z",
    "estimated_completion": "2024-12-21T01:30:00Z",
    "queue_position": 2,
    "resource_allocation_confirmed": true
  },
  "cost_estimate": {
    "compute_cost": "$12.50",
    "storage_cost": "$2.30",
    "total_estimated_cost": "$14.80"
  },
  "sla_commitment": {
    "max_completion_time": "2024-12-21T02:00:00Z",
    "availability_guarantee": "99.5%",
    "rollback_plan_available": true
  }
}
```

### 2. Comprehensive Job Status with Analytics

**Purpose**: Real-time job monitoring with predictive completion and resource analytics.

```bash
curl -X GET "http://localhost:3000/api/v1/reindex/jobs/job_550e8400-e29b-41d4-a716-446655440000"
```

Expected Response:
```json
{
  "job_id": "job_550e8400-e29b-41d4-a716-446655440000",
  "status": "InProgress",
  "current_stage": "IndexBuilding",
  "progress_analytics": {
    "overall_progress_percent": 62.3,
    "stage_progress_percent": 78.5,
    "documents_processed": 15575,
    "documents_remaining": 9425,
    "processing_rate_docs_per_sec": 1250
  },
  "time_analytics": {
    "started_at": "2024-12-20T22:00:00Z",
    "current_time": "2024-12-20T23:30:00Z",
    "elapsed_time_minutes": 90,
    "estimated_completion": "2024-12-21T01:15:00Z",
    "confidence_interval": "±15 minutes"
  },
  "resource_utilization": {
    "cpu_percent": 85.2,
    "memory_usage_gb": 9.8,
    "memory_peak_gb": 11.2,
    "disk_io_read_mbps": 450,
    "disk_io_write_mbps": 320,
    "network_io_mbps": 75
  },
  "performance_metrics": {
    "indexing_throughput": 1250,
    "memory_efficiency": 0.82,
    "cpu_efficiency": 0.91,
    "stage_breakdown": {
      "data_loading": "completed_15_min",
      "vector_processing": "completed_45_min", 
      "index_building": "in_progress_30_min",
      "validation": "pending",
      "deployment": "pending"
    }
  },
  "quality_metrics": {
    "index_integrity_check": "passing",
    "memory_leak_detection": "clean",
    "performance_regression_check": "within_tolerance"
  }
}
```

### 3. Job Queue Management with Priority Analytics

```bash
curl -X GET "http://localhost:3000/api/v1/reindex/jobs"
```

Expected Response:
```json
{
  "active_jobs": 2,
  "queued_jobs": 1,
  "completed_jobs_24h": 5,
  "jobs": [
    {
      "job_id": "job_550e8400",
      "collection_name": "company_documents",
      "status": "InProgress",
      "priority": "High",
      "progress_percent": 62.3,
      "estimated_completion": "2024-12-21T01:15:00Z"
    }
  ],
  "system_capacity": {
    "total_cpu_cores": 16,
    "available_cpu_cores": 4,
    "total_memory_gb": 64,
    "available_memory_gb": 18,
    "utilization_percent": 75
  },
  "queue_analytics": {
    "average_wait_time_minutes": 45,
    "peak_queue_length_24h": 8,
    "job_success_rate_percent": 98.2
  }
}
```

### 4. Advanced Job Control

```bash
# Pause job
curl -X POST "http://localhost:3000/api/v1/reindex/jobs/job_550e8400/pause"

# Resume job  
curl -X POST "http://localhost:3000/api/v1/reindex/jobs/job_550e8400/resume"

# Cancel with cleanup
curl -X DELETE "http://localhost:3000/api/v1/reindex/jobs/job_550e8400?cleanup=true"
```

## 📈 Day 9.3: Advanced Monitoring & Performance Intelligence

### 1. Multi-dimensional Health Assessment

**Purpose**: Comprehensive collection health analysis with predictive insights.

```bash
curl -X GET "http://localhost:3000/api/v1/monitor/health/company_documents"
```

Expected Response:
```json
{
  "collection_name": "company_documents",
  "overall_health": "Excellent",
  "health_score": 92,
  "last_updated": "2024-12-20T10:45:00Z",
  "dimensional_analysis": {
    "performance": {
      "grade": "A-",
      "query_latency_p50_ms": 12.3,
      "query_latency_p99_ms": 42.1,
      "throughput_queries_per_sec": 245
    },
    "reliability": {
      "grade": "A+",
      "uptime_percent_24h": 99.98,
      "error_rate_percent": 0.02,
      "failed_queries_24h": 3
    },
    "efficiency": {
      "grade": "B+",
      "memory_efficiency_percent": 82,
      "cpu_efficiency_percent": 78,
      "storage_efficiency_percent": 85
    },
    "scalability": {
      "grade": "A",
      "headroom_percent": 40,
      "projected_capacity_exhaustion": "6_months"
    }
  },
  "predictive_insights": {
    "performance_trend": "stable_with_slight_improvement",
    "capacity_forecast": "sufficient_for_3x_growth",
    "optimization_opportunities": [
      "Enable query result caching for 15% latency improvement",
      "Implement batch processing for 25% throughput gain"
    ]
  },
  "sla_compliance": {
    "latency_sla_99_percent": "compliant",
    "availability_sla": "compliant", 
    "throughput_sla": "exceeded_by_22_percent"
  }
}
```

### 2. Time-series Performance Analytics

**Purpose**: Historical performance analysis with trend detection and forecasting.

```bash
curl -X GET "http://localhost:3000/api/v1/monitor/metrics/company_documents?window=24h&granularity=5m"
```

Expected Response:
```json
{
  "collection_name": "company_documents",
  "time_range": {
    "start": "2024-12-19T10:45:00Z",
    "end": "2024-12-20T10:45:00Z",
    "granularity": "5m"
  },
  "metrics": {
    "query_latency_p99_ms": {
      "current": 42.1,
      "average": 38.5,
      "min": 28.2,
      "max": 67.8,
      "trend": "stable",
      "data_points": 288
    },
    "throughput_qps": {
      "current": 245,
      "average": 220,
      "peak": 412,
      "trend": "increasing_10_percent"
    },
    "memory_usage_gb": {
      "current": 2.8,
      "average": 2.6,
      "peak": 3.1,
      "growth_rate_per_day": 0.02
    }
  },
  "anomaly_detection": {
    "anomalies_detected": 2,
    "anomalies": [
      {
        "timestamp": "2024-12-20T03:15:00Z",
        "metric": "query_latency_p99_ms",
        "value": 67.8,
        "severity": "medium",
        "probable_cause": "batch_indexing_interference"
      }
    ]
  },
  "correlation_analysis": {
    "memory_vs_latency_correlation": 0.73,
    "throughput_vs_error_rate_correlation": -0.45,
    "insights": [
      "High memory usage strongly correlates with increased latency",
      "Higher throughput periods show lower error rates"
    ]
  }
}
```

### 3. Intelligent Alert System

**Purpose**: Proactive alert management with machine learning-based threshold optimization.

```bash
curl -X GET "http://localhost:3000/api/v1/monitor/alerts/company_documents"
```

Expected Response:
```json
{
  "active_alerts": [
    {
      "id": "alert_lat_001",
      "severity": "Warning",
      "metric_name": "query_latency_p99_ms",
      "current_value": 47.3,
      "threshold": 45.0,
      "message": "Query latency P99 exceeds optimized threshold",
      "triggered_at": "2024-12-20T10:35:00Z",
      "duration_minutes": 10,
      "impact_assessment": {
        "affected_queries_percent": 1,
        "user_experience_impact": "minimal",
        "business_impact": "low"
      },
      "suggested_actions": [
        "Monitor for 15 more minutes before intervention",
        "Consider scaling ef_search parameter",
        "Check for concurrent heavy operations"
      ]
    }
  ],
  "alert_analytics": {
    "alerts_24h": 12,
    "false_positive_rate": 0.05,
    "resolution_time_avg_minutes": 8,
    "auto_resolved_percent": 67
  },
  "threshold_optimization": {
    "last_optimization": "2024-12-18T10:45:00Z",
    "next_optimization": "2024-12-25T10:45:00Z",
    "ml_confidence": 0.91
  }
}
```

### 4. Advanced Alert Configuration with ML

```bash
curl -X POST "http://localhost:3000/api/v1/monitor/collections/company_documents/alerts" \
  -H "Content-Type: application/json" \
  -d '{
    "rules": [
      {
        "metric_name": "query_latency_p99_ms",
        "comparison": "GreaterThan",
        "threshold": 50.0,
        "severity": "Critical",
        "auto_optimization": {
          "enabled": true,
          "max_adjustments_per_day": 3,
          "ml_threshold_adjustment": true
        }
      },
      {
        "metric_name": "error_rate_percent", 
        "comparison": "GreaterThan",
        "threshold": 1.0,
        "severity": "Warning",
        "context_aware": {
          "business_hours_multiplier": 2.0,
          "correlate_with_deployment": true
        }
      }
    ],
    "notification_channels": [
      {
        "type": "webhook",
        "url": "https://your-monitoring.com/webhook",
        "severity_filter": ["Critical", "Warning"]
      },
      {
        "type": "email",
        "recipients": ["ops-team@company.com"],
        "severity_filter": ["Critical"]
      }
    ]
  }'
```

## 🧪 Complete Enterprise Testing Workflow

### End-to-End Performance Testing

```bash
#!/bin/bash
# Enterprise-grade testing workflow

echo "🚀 Starting comprehensive Day 9 testing workflow"

# 1. Register production-like collection
echo "📊 Registering collection with enterprise parameters..."
curl -X POST "http://localhost:3000/api/v1/index/collections/register" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "enterprise_docs",
    "vector_size": 1024,
    "current_doc_count": 1000000,
    "expected_growth_rate": 50000,
    "query_patterns": {
      "high_recall_queries": 40,
      "speed_critical_queries": 40,
      "balanced_queries": 20
    },
    "workload_characteristics": {
      "avg_queries_per_second": 1000,
      "peak_queries_per_second": 5000,
      "batch_indexing_frequency": "hourly"
    }
  }'

# 2. Establish baseline health metrics
echo "📈 Establishing baseline health metrics..."
curl -X GET "http://localhost:3000/api/v1/monitor/health/enterprise_docs" | jq .

# 3. Configure intelligent alerts
echo "🚨 Setting up intelligent alert system..."
curl -X POST "http://localhost:3000/api/v1/monitor/collections/enterprise_docs/alerts" \
  -H "Content-Type: application/json" \
  -d '{
    "rules": [
      {
        "metric_name": "query_latency_p99_ms",
        "comparison": "GreaterThan", 
        "threshold": 100.0,
        "severity": "Critical",
        "auto_optimization": {"enabled": true}
      }
    ]
  }'

# 4. Schedule optimization job
echo "⚙️ Scheduling background optimization..."
OPTIMIZATION_JOB=$(curl -X POST "http://localhost:3000/api/v1/reindex/schedule" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "enterprise_docs",
    "priority": "High",
    "optimization_target": "Balanced",
    "sla_requirements": {
      "max_duration_hours": 6,
      "availability_during_reindex": "ReadOnly"
    }
  }' | jq -r '.job_id')

echo "📋 Optimization job ID: $OPTIMIZATION_JOB"

# 5. Monitor job progress
echo "👁️ Monitoring optimization progress..."
while true; do
  STATUS=$(curl -s "http://localhost:3000/api/v1/reindex/jobs/$OPTIMIZATION_JOB" | jq -r '.status')
  PROGRESS=$(curl -s "http://localhost:3000/api/v1/reindex/jobs/$OPTIMIZATION_JOB" | jq -r '.progress_analytics.overall_progress_percent')
  
  echo "Status: $STATUS, Progress: $PROGRESS%"
  
  if [[ "$STATUS" == "Completed" || "$STATUS" == "Failed" ]]; then
    break
  fi
  
  sleep 30
done

# 6. Performance load testing
echo "🔥 Generating performance load..."
for i in {1..1000}; do
  curl -X POST "http://localhost:3000/api/v1/search" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"performance test query $i\", \"collection\": \"enterprise_docs\", \"limit\": 10}" &
    
  if (( i % 100 == 0 )); then
    echo "Generated $i queries..."
    wait # Wait for batch to complete
  fi
done

# 7. Analyze performance under load
echo "📊 Analyzing performance metrics under load..."
curl -X GET "http://localhost:3000/api/v1/monitor/metrics/enterprise_docs?window=5m" | jq '.metrics'

# 8. Check for triggered alerts
echo "🚨 Checking alert status..."
curl -X GET "http://localhost:3000/api/v1/monitor/alerts/enterprise_docs" | jq '.active_alerts'

# 9. Get optimization recommendations
echo "💡 Getting post-load optimization recommendations..."
curl -X GET "http://localhost:3000/api/v1/index/collections/enterprise_docs/recommendations" | jq '.recommendations'

echo "✅ Enterprise testing workflow completed!"
```

### Performance Benchmarking

```bash
# Latency benchmarking
echo "⚡ Latency Benchmark Test"
for concurrency in 1 10 50 100; do
  echo "Testing concurrency: $concurrency"
  
  # Generate concurrent load
  for i in $(seq 1 $concurrency); do
    (
      for j in {1..100}; do
        curl -w "%{time_total}\n" -o /dev/null -s \
          -X POST "http://localhost:3000/api/v1/search" \
          -H "Content-Type: application/json" \
          -d '{"query": "benchmark test", "collection": "enterprise_docs", "limit": 10}'
      done
    ) &
  done
  
  wait
  
  # Get metrics
  curl -X GET "http://localhost:3000/api/v1/monitor/metrics/enterprise_docs?window=1m" | \
    jq '.metrics.query_latency_p99_ms'
done
```

## 📊 Business Intelligence Dashboards

### Key Performance Indicators (KPIs)

```bash
# Generate KPI report
curl -X GET "http://localhost:3000/api/v1/monitor/kpi/enterprise_docs" \
  -H "Accept: application/json"
```

Expected KPI Response:
```json
{
  "collection_name": "enterprise_docs",
  "reporting_period": "24h",
  "business_kpis": {
    "availability_sla": {
      "target": 99.9,
      "actual": 99.95,
      "status": "exceeding"
    },
    "performance_sla": {
      "target_p99_ms": 100,
      "actual_p99_ms": 67,
      "status": "exceeding"
    },
    "cost_efficiency": {
      "cost_per_million_queries": 2.45,
      "vs_baseline": "-15%",
      "optimization_savings": "$1,250/month"
    },
    "user_satisfaction": {
      "derived_satisfaction_score": 4.7,
      "query_success_rate": 99.8,
      "user_retention_impact": "+12%"
    }
  },
  "operational_kpis": {
    "system_utilization": 72,
    "auto_scaling_efficiency": 89,
    "maintenance_overhead_hours": 2.5
  }
}
```

This comprehensive testing suite demonstrates the enterprise-grade capabilities of your Day 9 vector database optimization system, providing both technical depth and business intelligence insights.