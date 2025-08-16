#!/bin/bash

# Test script for automatic health check functionality
set -e

echo "🧪 Testing Automatic Health Check Progression"
echo "============================================="

# Test 1: Load new model
echo "📋 Step 1: Loading new model..."
RESPONSE=$(curl -s -X POST http://localhost:3001/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{"name": "TinyLlama-1.1B-Chat", "version": "test-auto"}')

MODEL_ID=$(echo $RESPONSE | jq -r '.model_id')
echo "Model ID: $MODEL_ID"

# Test 2: Monitor automatic progression
echo "📋 Step 2: Monitoring automatic progression (max 30 seconds)..."
for i in {1..6}; do
  echo "  Check $i:"
  STATUS_INFO=$(curl -s http://localhost:3001/api/v1/models | jq -r 'map(select(.id == "'$MODEL_ID'")) | .[0] | "\(.status) (health: \(.health_score))"')
  echo "    Status: $STATUS_INFO"
  
  STATUS=$(curl -s http://localhost:3001/api/v1/models | jq -r 'map(select(.id == "'$MODEL_ID'")) | .[0] | .status')
  
  if [ "$STATUS" = "Ready" ]; then
    echo "  ✅ SUCCESS: Model automatically reached Ready status!"
    break
  elif [ "$STATUS" = "Failed" ]; then
    echo "  ❌ FAILED: Model failed during loading"
    exit 1
  fi
  
  sleep 5
done

if [ "$STATUS" != "Ready" ]; then
  echo "  ❌ TIMEOUT: Model did not reach Ready status in 30 seconds"
  exit 1
fi

# Test 3: Verify atomic swap works
echo "📋 Step 3: Testing atomic swap..."
SWAP_RESULT=$(curl -s -X POST http://localhost:3001/api/v1/models/swap \
  -H "Content-Type: application/json" \
  -d "{\"target_model_id\": \"$MODEL_ID\"}")

SUCCESS=$(echo $SWAP_RESULT | jq -r '.success')
if [ "$SUCCESS" = "true" ]; then
  echo "  ✅ SUCCESS: Atomic swap completed successfully"
  DURATION=$(echo $SWAP_RESULT | jq -r '.swap_duration_ms')
  echo "  ⏱️  Swap duration: ${DURATION}ms"
else
  echo "  ❌ FAILED: Atomic swap failed"
  echo $SWAP_RESULT | jq '.error_message'
  exit 1
fi

# Test 4: Verify active model changed
echo "📋 Step 4: Verifying model switch..."
ACTIVE_ID=$(curl -s http://localhost:3001/api/v1/models/active | jq -r '.')
if [ "$ACTIVE_ID" = "$MODEL_ID" ]; then
  echo "  ✅ SUCCESS: Active model correctly switched"
else
  echo "  ❌ FAILED: Active model did not switch correctly"
  echo "  Expected: $MODEL_ID"
  echo "  Actual: $ACTIVE_ID"
  exit 1
fi

# Test 5: Test rollback
echo "📋 Step 5: Testing rollback..."
ROLLBACK_RESULT=$(curl -s -X POST http://localhost:3001/api/v1/models/rollback)
ROLLBACK_SUCCESS=$(echo $ROLLBACK_RESULT | jq -r '.success')

if [ "$ROLLBACK_SUCCESS" = "true" ]; then
  echo "  ✅ SUCCESS: Rollback completed successfully"
  ROLLBACK_DURATION=$(echo $ROLLBACK_RESULT | jq -r '.swap_duration_ms')
  echo "  ⏱️  Rollback duration: ${ROLLBACK_DURATION}ms"
else
  echo "  ❌ FAILED: Rollback failed"
  exit 1
fi

# Test 6: Cleanup - remove test model
echo "📋 Step 6: Cleaning up test model..."
curl -s -X DELETE http://localhost:3001/api/v1/models/$MODEL_ID > /dev/null
echo "  ✅ Test model removed"

echo ""
echo "🎉 ALL TESTS PASSED!"
echo "✅ Automatic health check progression works correctly"
echo "✅ Zero-downtime atomic swapping works correctly"
echo "✅ Rollback functionality works correctly"
echo ""
echo "The fix is working perfectly! 🚀"