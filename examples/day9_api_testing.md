# Day 9 API Testing Examples

This document demonstrates how to test the Day 9.1-9.3 index management features via REST APIs.

## Prerequisites

Start the server:
```bash
cargo run
```

The server will be available at `http://localhost:8080`

## Day 9.1: Index Configuration Optimization

### 1. Register Collection for Optimization

```bash
curl -X POST "http://localhost:8080/api/v1/index/collections/register" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents",
    "vector_size": 384,
    "current_doc_count": 10000,
    "query_patterns": {
      "high_recall_queries": 70,
      "speed_critical_queries": 20,
      "balanced_queries": 10
    }
  }'
```

Expected Response:
```json
{
  "message": "Collection registered successfully",
  "recommended_profile": "HighAccuracy",
  "config": {
    "hnsw_config": {
      "m": 64,
      "ef_construct": 200,
      "max_indexing_threads": 4
    },
    "quantization": null
  }
}
```

### 2. Get Optimization Recommendations

```bash
curl -X GET "http://localhost:8080/api/v1/index/collections/documents/recommendations"
```

### 3. Apply Optimization Profile

```bash
curl -X POST "http://localhost:8080/api/v1/index/collections/documents/apply-profile" \
  -H "Content-Type: application/json" \
  -d '{
    "profile": "FastQuery"
  }'
```

## Day 9.2: Background Reindexing

### 1. Schedule Background Reindex Job

```bash
curl -X POST "http://localhost:8080/api/v1/reindex/schedule" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents",
    "priority": "High",
    "estimated_duration_hours": 2.5,
    "resource_requirements": {
      "cpu_cores": 4,
      "memory_gb": 8.0,
      "disk_io_mbps": 500
    },
    "optimization_target": "HighAccuracy"
  }'
```

Expected Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Reindex job scheduled successfully",
  "estimated_start_time": "2024-12-20T10:30:00Z"
}
```

### 2. Check Job Status

```bash
curl -X GET "http://localhost:8080/api/v1/reindex/jobs/550e8400-e29b-41d4-a716-446655440000"
```

Expected Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "InProgress",
  "state": "IndexBuilding",
  "progress": 45.5,
  "stage": "Building HNSW index",
  "started_at": "2024-12-20T10:30:00Z",
  "estimated_completion": "2024-12-20T13:00:00Z",
  "resource_usage": {
    "cpu_percent": 75.2,
    "memory_gb": 6.8,
    "disk_io_mbps": 450
  }
}
```

### 3. List All Active Jobs

```bash
curl -X GET "http://localhost:8080/api/v1/reindex/jobs"
```

### 4. Cancel Job (if needed)

```bash
curl -X DELETE "http://localhost:8080/api/v1/reindex/jobs/550e8400-e29b-41d4-a716-446655440000"
```

## Day 9.3: Index Monitoring and Performance Metrics

### 1. Check Collection Health

```bash
curl -X GET "http://localhost:8080/api/v1/monitor/health/documents"
```

Expected Response:
```json
{
  "collection_name": "documents",
  "overall_health": "Good",
  "last_updated": "2024-12-20T10:45:00Z",
  "metrics": {
    "query_latency_p99_ms": 45.2,
    "index_memory_usage_mb": 512.8,
    "indexing_rate_docs_per_sec": 1250.0,
    "error_rate_percent": 0.1
  },
  "performance_grade": "B+",
  "recommendations": [
    "Consider increasing ef parameter for better recall"
  ]
}
```

### 2. Get Performance Metrics

```bash
curl -X GET "http://localhost:8080/api/v1/monitor/metrics/documents?window=1h"
```

### 3. Get Current Alerts

```bash
curl -X GET "http://localhost:8080/api/v1/monitor/alerts/documents"
```

Expected Response:
```json
{
  "active_alerts": [
    {
      "id": "alert_001",
      "severity": "Warning",
      "metric_name": "query_latency_p99_ms",
      "current_value": 85.2,
      "threshold": 80.0,
      "message": "Query latency P99 exceeds threshold",
      "triggered_at": "2024-12-20T10:40:00Z"
    }
  ],
  "alert_summary": {
    "critical": 0,
    "warning": 1,
    "info": 0
  }
}
```

### 4. Configure Alert Rules

```bash
curl -X POST "http://localhost:8080/api/v1/monitor/collections/documents/alerts" \
  -H "Content-Type: application/json" \
  -d '{
    "rules": [
      {
        "metric_name": "query_latency_p99_ms",
        "comparison": "GreaterThan",
        "threshold": 100.0,
        "severity": "Critical"
      },
      {
        "metric_name": "error_rate_percent",
        "comparison": "GreaterThan",
        "threshold": 5.0,
        "severity": "Warning"
      }
    ]
  }'
```

## Complete Testing Workflow

Here's a complete workflow to test all Day 9 features:

```bash
# 1. Register a collection
curl -X POST "http://localhost:8080/api/v1/index/collections/register" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "test_docs",
    "vector_size": 384,
    "current_doc_count": 5000,
    "query_patterns": {"balanced_queries": 100}
  }'

# 2. Check initial health
curl -X GET "http://localhost:8080/api/v1/monitor/health/test_docs"

# 3. Schedule optimization job
curl -X POST "http://localhost:8080/api/v1/reindex/schedule" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "test_docs",
    "priority": "Medium",
    "estimated_duration_hours": 1.0,
    "optimization_target": "Balanced"
  }'

# 4. Monitor job progress
curl -X GET "http://localhost:8080/api/v1/reindex/jobs"

# 5. Check performance after optimization
curl -X GET "http://localhost:8080/api/v1/monitor/metrics/test_docs?window=30m"
```

## Performance Testing

Use these commands to generate load and observe the monitoring system:

```bash
# Generate query load (requires existing vectors)
for i in {1..100}; do
  curl -X POST "http://localhost:8080/api/v1/search" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"test query $i\", \"collection\": \"test_docs\", \"limit\": 10}" &
done

# Monitor performance during load
curl -X GET "http://localhost:8080/api/v1/monitor/metrics/test_docs?window=5m"
```