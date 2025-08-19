 🚀 Complete Guide: Starting AI Inference Server with Qdrant and Testing APIs

  📋 Prerequisites

  Make sure you have the following installed:
  - Docker & Docker Compose
  - Rust & Cargo (latest stable)
  - curl (for API testing)

  🐳 Step 1: Start Qdrant Vector Database

  Option A: Using Docker Compose (Recommended)

  # Navigate to project directory
  cd /Users/abhinawkumar/Code/ai-cloud-native/ai-interence-server

 docker stop qdrant-dev  
   docker rm qdrant-dev      
  # Start Qdrant with production configuration
  docker-compose -f docker-compose.qdrant.yml up -d qdrant

  # Check if Qdrant is running
  docker ps | grep qdrant

  # Verify Qdrant health
  curl http://localhost:6333
  # Expected: {"title":"qdrant - vector search engine","version":"1.7.4"}

  Option B: Simple Docker Run

  # Alternative simple start
  docker run -d --name qdrant-dev -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.7.4

  # Check health
  curl http://localhost:6333/collections
  # Expected: {"result":{"collections":[]},"status":"ok","time":...}

  ⚙️ Step 2: Environment Configuration (Optional)

  # Set environment variables for optimal performance
  export BATCH_MAX_SIZE=4
  export BATCH_MAX_WAIT_MS=100
  export BATCH_MAX_QUEUE_SIZE=50
  export RAYON_NUM_THREADS=4
  export TOKENIZERS_PARALLELISM=false

  # Qdrant configuration (if needed)
  export QDRANT_URL=http://localhost:6333
  export QDRANT_COLLECTION_NAME=conversations

  🚀 Step 3: Start AI Inference Server

  # Navigate to project directory
  cd /Users/abhinawkumar/Code/ai-cloud-native/ai-interence-server

  # Start the server
  cargo run

  Expected Startup Logs:

  🚀 Starting AI Inference Server with Batching
  📊 Batch Configuration - Max Size: 4, Max Wait: 100ms, Max Queue: 50
  🗂️ Initializing vector storage with smart backend selection...
  🏭 Initializing vector storage factory...
  🔗 Initializing Qdrant client: http://localhost:6333
  ✅ Using qdrant vector storage backend  # <-- Success!
  📦 Loading initial TinyLlama model version...
  🚀 Metal GPU available, using acceleration
  ✅ Model loaded successfully
  🌐 Server starting on http://0.0.0.0:3000
  ✅ Server ready and accepting requests

  🧪 Step 4: Test All APIs

  4.1 Health Check

  curl http://localhost:3000/health
  # Expected: 200 OK

  4.2 Vector Operations

  Insert Vector

  curl -X POST http://localhost:3000/api/v1/vectors \
    -H "Content-Type: application/json" \
    -d '{
      "vector": [1.0, 0.5, 0.8, 0.2, 0.9],
      "metadata": {
        "type": "test_document",
        "content": "This is a test document for vector search",
        "category": "testing"
      }
    }'

  # Expected Response:
  # {"id":"uuid-here","success":true,"message":"Vector inserted successfully"}

  Search Similar Vectors

  curl -X POST http://localhost:3000/api/v1/vectors/search \
    -H "Content-Type: application/json" \
    -d '{
      "vector": [1.0, 0.5, 0.8, 0.2, 0.9],
      "limit": 5
    }'

  # Expected Response:
  # {"results":[{"id":"uuid","similarity":1.0,"metadata":{...}}],"total_found":1}

  Get Vector by ID

  # Use the ID from insert response
  VECTOR_ID="your-vector-id-here"
  curl -X GET http://localhost:3000/api/v1/vectors/$VECTOR_ID

  # Expected Response:
  # {"id":"uuid","vector":[1.0,0.5,0.8,0.2,0.9],"metadata":{...}}

  Vector Storage Statistics

  curl -X GET http://localhost:3000/api/v1/vectors/stats

  # Expected Response:
  # {"total_vectors":1,"memory_usage_estimate":150}

  List All Vectors

  curl -X GET http://localhost:3000/api/v1/vectors/list

  # Expected Response:
  # [{"id":"uuid","vector":[...],"metadata":{...}}]

  4.3 AI Text Generation with Memory

  Generate Text with Conversation Memory

  curl -X POST http://localhost:3000/api/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "Tell me a short joke about programming",
      "max_tokens": 50,
      "use_memory": true,
      "memory_limit": 3
    }'

  # Expected Response includes:
  # {
  #   "text": "Generated text...",
  #   "memory_used": true,
  #   "context_retrieved": 0,  // First conversation
  #   "tokens_per_second": 15.5,
  #   ...
  # }

  Follow-up with Memory

  curl -X POST http://localhost:3000/api/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "What did I just ask you about?",
      "max_tokens": 30,
      "use_memory": true
    }'

  # Expected Response includes:
  # {
  #   "text": "You asked about programming jokes...",
  #   "memory_used": true,
  #   "context_retrieved": 1,  // Retrieved previous conversation
  #   ...
  # }

  4.4 Model Management APIs

  List Models

  curl -X GET http://localhost:3000/api/v1/models

  # Expected Response:
  # {"models":[{"id":"uuid","name":"TinyLlama-1.1B-Chat","status":"Active"}]}

  Get System Status

  curl -X GET http://localhost:3000/api/v1/system/status

  # Expected Response:
  # {"active_model":"TinyLlama-1.1B-Chat","total_models":1,"memory_usage":...}

  Batch Processing Status

  curl -X GET http://localhost:3000/api/v1/generate/status

  # Expected Response:
  # {"queue_size":0,"total_requests":2,"avg_processing_time_ms":1500}

  🔍 Step 5: Verify Qdrant Integration

  Check Qdrant Collections

  curl http://localhost:6333/collections

  # Should show the conversation collection:
  # {"result":{"collections":[{"name":"conversations"}]},"status":"ok"}

  Check Collection Details

  curl http://localhost:6333/collections/conversations

  # Shows collection info with vector count

  Qdrant Web Dashboard

  Open in browser: http://localhost:6333/dashboard

  📊 Step 6: Performance Testing

  Load Testing Script

  # Create a simple load test
  for i in {1..5}; do
    echo "Request $i:"
    curl -X POST http://localhost:3000/api/v1/generate \
      -H "Content-Type: application/json" \
      -d "{\"prompt\":\"Test request $i\",\"max_tokens\":20,\"
      use_memory\":true}" \
      -w "Time: %{time_total}s\n"
    echo "---"
  done

  🛠️ Troubleshooting

  If Qdrant Connection Fails

  The system will automatically fall back to in-memory storage:
  ⚠️ Qdrant unavailable (...), falling back to in-memory storage
  ✅ Using in-memory vector storage backend

  Check Logs

  # Server logs show detailed information
  # Look for:
  # - "✅ Using qdrant vector storage backend" (Success)
  # - "✅ Using in-memory vector storage backend" (Fallback)

  # Qdrant logs
  docker logs qdrant-dev

  Restart Services

  # Restart Qdrant
  docker-compose -f docker-compose.qdrant.yml restart qdrant

  # Or restart everything
  docker-compose -f docker-compose.qdrant.yml down
  docker-compose -f docker-compose.qdrant.yml up -d

  📈 Expected Performance Metrics

  - Text Generation: 15-20 tokens/second with Metal GPU
  - Vector Search: Sub-millisecond similarity search
  - Memory Integration: Context retrieval in <50ms
  - API Response: <100ms for vector operations

  🎯 Success Indicators

  ✅ Qdrant Connected: Server logs show "Using qdrant vector storage backend"✅ Model Loaded: TinyLlama loads with Metal GPU
  acceleration✅ Vector Operations: All CRUD operations work via REST APIs✅ Memory Integration: Conversation context stored and
   retrieved✅ Performance: Good tokens/second and low latency

  🧹 Cleanup (When Done)

  # Stop AI server (Ctrl+C)

  # Stop Qdrant
  docker-compose -f docker-compose.qdrant.yml down

  # Or remove everything including volumes
  docker-compose -f docker-compose.qdrant.yml down -v

  This complete setup gives you a production-ready AI inference server with Qdrant vector database integration, conversation
  memory, and comprehensive API testing capabilities!