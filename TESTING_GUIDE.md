# 🧪 **Complete Testing Guide for Version Management & Atomic Swap**

## **🚀 Prerequisites**

```bash
# Ensure you have the latest code
git status
cargo build --release

# Optional: Set environment variables
export RUST_LOG=info
export PORT=3001
```

---

## **📋 Test Suite 1: Basic Functionality**

### **Step 1: Server Startup Performance Test**

```bash
# Terminal 1: Start server with timing
echo "Starting server at $(date)"
time RUST_LOG=info ./target/release/ai-interence-server
```

**Expected Output:**
```
🚀 Starting AI Inference Server with Batching
📊 Batch Configuration - Max Size: 1, Max Wait: 5ms, Max Queue: 4
🚀 Initializing Model Version Manager...
📦 Loading initial TinyLlama model version...
⏳ Waiting for model to load...
Model still loading, attempt 1/15...
🏥 Health check result: score 1.00
✅ Model [uuid] activated successfully
⚛️ Atomic model swap system initialized
🌐 Server starting on http://0.0.0.0:3001
✅ Server ready and accepting requests
```

**⏱️ Expected Timing:** 5-8 seconds total startup

---

### **Step 2: Health Check Validation**

```bash
# Terminal 2: Test health endpoint
curl -s http://localhost:3001/health | jq

# Expected: {"status": "healthy"}
```

### **Step 3: Model Management API Tests**

```bash
# List all models
curl -s http://localhost:3001/api/v1/models | jq

# Get active model ID  
curl -s http://localhost:3001/api/v1/models/active

# Get system status
curl -s http://localhost:3001/api/v1/system/status | jq
```

**Expected Output:**
```json
[
  {
    "id": "uuid-here",
    "name": "TinyLlama-1.1B-Chat", 
    "version": "v1.0",
    "status": "Active",
    "health_score": 1.0
  }
]
```

---

## **⚡ Test Suite 2: Inference Performance**

### **Step 4: Baseline Inference Tests**

```bash
# Test 1: Short generation
curl -s -X POST http://localhost:3001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 20}' | jq -r '
    "Tokens/sec: " + (.tokens_per_second | tostring) +
    " | Processing: " + (.processing_time_ms | tostring) + "ms" +  
    " | Queue: " + (.queue_time_ms | tostring) + "ms" +
    " | Text: " + .text'
```

```bash
# Test 2: Medium generation
curl -s -X POST http://localhost:3001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain machine learning in simple terms", "max_tokens": 50}' | jq -r '
    "Tokens/sec: " + (.tokens_per_second | tostring) +
    " | Processing: " + (.processing_time_ms | tostring) + "ms"'
```

```bash
# Test 3: Concurrent requests (run multiple times)
for i in {1..3}; do
  curl -s -X POST http://localhost:3001/api/v1/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Test '"$i"': What is AI?", "max_tokens": 15}' | jq -r '.tokens_per_second' &
done
wait
```

**📊 Expected Performance:**
- First request: 5-10 tok/s (cold start)
- Subsequent requests: 10-20 tok/s
- Queue time: 2-10ms

---

## **🔄 Test Suite 3: Hot Model Swapping**

### **Step 5: Load Second Model**

```bash
# Load new model version
echo "Loading second model at $(date)"
curl -s -X POST http://localhost:3001/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{"name": "TinyLlama-1.1B-Chat", "version": "v2.0"}' | jq

# Model will be loaded in background with automatic health check
```

### **Step 6: Monitor Loading Progress**

```bash
# Check loading status every 5 seconds (automatic health check now included)
for i in {1..8}; do
  echo "Check $i at $(date):"
  curl -s http://localhost:3001/api/v1/models | jq 'map(select(.version == "v2.0")) | .[0] | {version, status, health_score}'
  
  # Check if model is ready
  STATUS=$(curl -s http://localhost:3001/api/v1/models | jq -r 'map(select(.version == "v2.0")) | .[0] | .status // "not_found"')
  if [ "$STATUS" = "Ready" ]; then
    echo "✅ Model is ready!"
    break
  elif [ "$STATUS" = "Failed" ]; then
    echo "❌ Model failed to load"
    break
  fi
  
  sleep 5
done

# Get the model ID for the ready model
MODEL_2_ID=$(curl -s http://localhost:3001/api/v1/models | jq -r '.[] | select(.version == "v2.0" and .status == "Ready") | .id')
echo "Model v2.0 ID: $MODEL_2_ID"
```

**Expected Progression (with automatic health check):**
```
Check 1: {"version": "v2.0", "status": "Loading", "health_score": 0.0}
Check 2: {"version": "v2.0", "status": "Loading", "health_score": 0.0}  
Check 3: {"version": "v2.0", "status": "HealthCheck", "health_score": 0.0}
Check 4: {"version": "v2.0", "status": "HealthCheck", "health_score": 0.0} (running health check)
Check 5: {"version": "v2.0", "status": "Ready", "health_score": 1.0} ✅ AUTOMATIC!
```

---

### **Step 7: Perform Atomic Swap**

```bash
# Verify model is ready before swap
echo "Verifying model readiness:"
curl -s http://localhost:3001/api/v1/models | jq 'map(select(.id == "'$MODEL_2_ID'")) | .[0] | {status, health_score}'

# Perform atomic swap with timing
echo "Performing atomic swap at $(date)"
time curl -s -X POST http://localhost:3001/api/v1/models/swap \
  -H "Content-Type: application/json" \
  -d "{\"target_model_id\": \"$MODEL_2_ID\"}" | jq '.success, .swap_duration_ms, .health_check.overall_score'
```

**✅ Expected Result:**
```
true
2000-3000
1.0
```

### **Step 8: Verify Zero-Downtime Swap**

```bash
# CRITICAL TEST: Run this DURING the swap to verify no downtime
# Terminal 3: Start continuous requests
for i in {1..20}; do
  curl -s -X POST http://localhost:3001/api/v1/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Request '$i' during swap", "max_tokens": 10}' | jq -r '.text' &
  sleep 0.5
done
wait

# Terminal 2: During the above loop, perform the swap
MODEL_2_ID=$(curl -s http://localhost:3001/api/v1/models | jq -r '.[] | select(.status == "Ready") | .id')
curl -s -X POST http://localhost:3001/api/v1/models/swap \
  -H "Content-Type: application/json" \
  -d "{\"target_model_id\": \"$MODEL_2_ID\"}"
```

**✅ Success Criteria:** All 20 requests should complete successfully with no errors

---

### **Step 9: Verify Model Switch**

```bash
# Check active model changed
echo "Active model after swap:"
curl -s http://localhost:3001/api/v1/models/active

# Check model statuses
curl -s http://localhost:3001/api/v1/models | jq 'map({version, status})'

# Test inference with new model
curl -s -X POST http://localhost:3001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Testing new model:", "max_tokens": 15}' | jq -r '.text'
```

**Expected:**
```
"new-model-uuid-here"
[
  {"version": "v1.0", "status": "Deprecated"},
  {"version": "v2.0", "status": "Active"}  
]
```

---

## **🔙 Test Suite 4: Rollback Testing**

### **Step 10: Test Rollback Functionality**

```bash
# Perform rollback
echo "Testing rollback at $(date)"
time curl -s -X POST http://localhost:3001/api/v1/models/rollback | jq '.success, .swap_duration_ms'

# Verify rollback worked  
echo "Active model after rollback:"
curl -s http://localhost:3001/api/v1/models/active

curl -s http://localhost:3001/api/v1/models | jq 'map({version, status})'
```

**Expected:**
```
true
8000-12000
[
  {"version": "v1.0", "status": "Active"},
  {"version": "v2.0", "status": "Deprecated"}
]
```

---

## **📊 Test Suite 5: Performance Analysis**

### **Step 11: Memory Usage Analysis**

```bash
# Get server process ID
SERVER_PID=$(pgrep ai-interence-server)
echo "Server PID: $SERVER_PID"

# Monitor memory usage (macOS)
top -pid $SERVER_PID -l 1 -stats pid,command,mem,rsize

# Or on Linux:
# ps -p $SERVER_PID -o pid,vsz,rss,pmem,comm
```

### **Step 12: Load Testing**

```bash
# Create a simple load test script
cat > load_test.sh << 'EOF'
#!/bin/bash
for i in {1..10}; do
  curl -s -X POST http://localhost:3001/api/v1/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Load test request '$i'", "max_tokens": 20}' | jq -r '.tokens_per_second' &
done
wait
EOF

chmod +x load_test.sh
time ./load_test.sh
```

### **Step 13: Advanced Model Management Tests**

```bash
# Test safety validation
MODEL_ID=$(curl -s http://localhost:3001/api/v1/models | jq -r '.[0].id')
curl -s http://localhost:3001/api/v1/models/$MODEL_ID/swap/safety | jq '.is_safe, .safety_checks[].passed'

# Test health check endpoint
curl -s -X POST http://localhost:3001/api/v1/models/$MODEL_ID/health | jq '.overall_score, .checks[] | {name, status, score}'

# Test model removal (be careful!)
# curl -s -X DELETE http://localhost:3001/api/v1/models/$MODEL_ID

# Get detailed system status
curl -s http://localhost:3001/api/v1/system/status | jq
```

---

## **🎯 Expected Performance Benchmarks**

| **Test** | **Expected Result** | **Pass Criteria** |
|----------|-------------------|-------------------|
| **Startup Time** | 5-8 seconds | < 10 seconds |
| **First Inference** | 5-10 tok/s | > 3 tok/s |
| **Warm Inference** | 10-20 tok/s | > 8 tok/s |
| **Model Load** | 8-12 seconds | < 15 seconds |
| **Atomic Swap** | 2-4 seconds | < 5 seconds |
| **Rollback** | 8-12 seconds | < 15 seconds |
| **Zero Downtime** | 0 failed requests | 100% success |
| **Memory (2 models)** | 6-10GB | System dependent |

---

## **🚨 Troubleshooting Guide**

### **Common Issues:**

1. **Server fails to start:**
   ```bash
   # Check logs
   RUST_LOG=debug ./target/release/ai-interence-server
   ```

2. **Model loading timeout:**
   ```bash
   # Check available memory
   free -h  # Linux
   vm_stat  # macOS
   ```

3. **Swap fails:**
   ```bash
   # Check model status
   curl -s http://localhost:3001/api/v1/models | jq 'map({id, status, health_score})'
   ```

4. **High memory usage:**
   ```bash
   # Monitor during operation
   watch -n 1 'ps aux | grep ai-interence-server | grep -v grep'
   ```

### **Performance Tuning:**

```bash
# Optimize batch configuration
export BATCH_MAX_SIZE=4
export BATCH_MAX_WAIT_MS=100

# Reduce memory pressure
export MAX_MODELS=2

# Enable verbose logging
export RUST_LOG=debug
```

---

## **🎯 Quick Test Script**

For convenience, here's a complete automated test script:

```bash
#!/bin/bash

# Quick Test Script for Version Management
set -e

echo "🧪 Starting AI Inference Server Version Management Tests"
echo "======================================================"

# Step 1: Start server in background
echo "📋 Step 1: Starting server..."
RUST_LOG=warn ./target/release/ai-interence-server &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for startup
sleep 10

# Step 2: Basic health check
echo "📋 Step 2: Health check..."
curl -s http://localhost:3001/health | jq

# Step 3: Inference test
echo "📋 Step 3: Baseline inference test..."
curl -s -X POST http://localhost:3001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}' | jq -r '.tokens_per_second'

# Step 4: Load second model
echo "📋 Step 4: Loading second model..."
curl -s -X POST http://localhost:3001/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{"name": "TinyLlama-1.1B-Chat", "version": "v2.0"}' | jq '.model_id'

# Wait for model loading
sleep 15

# Step 5: Atomic swap test
echo "📋 Step 5: Testing atomic swap..."
MODEL_2_ID=$(curl -s http://localhost:3001/api/v1/models | jq -r '.[] | select(.version == "v2.0") | .id')
curl -s -X POST http://localhost:3001/api/v1/models/swap \
  -H "Content-Type: application/json" \
  -d "{\"target_model_id\": \"$MODEL_2_ID\"}" | jq '.success'

# Step 6: Rollback test
echo "📋 Step 6: Testing rollback..."
curl -s -X POST http://localhost:3001/api/v1/models/rollback | jq '.success'

# Cleanup
echo "🧹 Cleaning up..."
kill $SERVER_PID

echo "✅ Tests completed!"
```

Save this as `quick_test.sh` and run with:
```bash
chmod +x quick_test.sh
./quick_test.sh
```

This comprehensive testing guide will help you verify all aspects of your version management and atomic swap implementation!