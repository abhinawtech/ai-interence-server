# 🧪 **VECTOR DATABASE TESTING GUIDE**

## 🚀 **SETUP INSTRUCTIONS**

### 1. **Environment Setup**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# For local testing, the defaults should work
```

### 2. **Start Qdrant Vector Database (Required)**
```bash
# Option A: Using Docker Compose (Recommended)
docker-compose -f docker-compose.qdrant.yml up -d

# Option B: Using Docker directly
docker run -p 6333:6333 qdrant/qdrant:latest

# Verify Qdrant is running
curl http://localhost:6333/collections
```

### 3. **Start the AI Inference Server**
```bash
# Build and run the server
cargo run

# You should see logs indicating:
# ✅ Qdrant client initialized successfully
# ✅ Vector collections initialized
# ✅ Embedding processor initialized
```

---

## 📊 **BEFORE vs AFTER: API COMPARISON**

### **🔸 BEFORE: Original API Endpoints**
```bash
# Basic health check
GET /health

# Text generation
POST /api/v1/generate

# Model management
GET /api/v1/models
POST /api/v1/models
GET /api/v1/system/status
```

### **🔸 AFTER: Enhanced API with Vector Operations**
```bash
# 🔷 ORIGINAL ENDPOINTS (Enhanced)
GET /health                          # Now includes vector DB health
POST /api/v1/generate               # Can now store embeddings automatically
GET /api/v1/models                  # Enhanced with vector integration status

# 🔷 NEW VECTOR DATABASE ENDPOINTS
POST /api/v1/vectors                # Insert single vector
POST /api/v1/vectors/batch          # Batch vector insertion  
POST /api/v1/vectors/search         # Similarity search
PUT /api/v1/vectors/:id            # Update vector
DELETE /api/v1/vectors/:id         # Delete vector

# 🔷 NEW COLLECTION MANAGEMENT
POST /api/v1/collections           # Create collection
GET /api/v1/collections            # List collections
GET /api/v1/collections/:name      # Collection info

# 🔷 NEW MONITORING ENDPOINTS
GET /api/v1/vectors/health         # Vector DB health
GET /api/v1/vectors/metrics        # Vector operations metrics
```

---

## 🧪 **STEP-BY-STEP TESTING SCENARIOS**

### **TEST 1: Basic Health Check (Before vs After)**

**🔸 BEFORE:**
```bash
curl http://localhost:3000/health
# Response: {"status": "healthy"}
```

**🔸 AFTER:**
```bash
curl http://localhost:3000/health
# Response: Enhanced with vector DB status

curl http://localhost:3000/api/v1/vectors/health
# New endpoint showing vector DB health
```

### **TEST 2: Vector Database Operations**

**Step 1: Check Collections**
```bash
curl http://localhost:3000/api/v1/collections
# Should show: ai_embeddings, conversations
```

**Step 2: Insert Your First Vector**
```bash
curl -X POST http://localhost:3000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "metadata": {
      "title": "Test Document",
      "category": "example",
      "created_by": "user_test"
    },
    "collection": "ai_embeddings"
  }'
```

**Step 3: Search for Similar Vectors**
```bash
curl -X POST http://localhost:3000/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "limit": 5,
    "score_threshold": 0.7,
    "include_vector": true,
    "collection": "ai_embeddings"
  }'
```

**Step 4: Batch Insert Multiple Vectors**
```bash
curl -X POST http://localhost:3000/api/v1/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {
        "vector": [0.2, 0.3, 0.4, 0.5, 0.6],
        "metadata": {"title": "Document 1", "category": "tech"}
      },
      {
        "vector": [0.3, 0.4, 0.5, 0.6, 0.7],
        "metadata": {"title": "Document 2", "category": "science"}
      }
    ],
    "collection": "ai_embeddings",
    "batch_size": 10
  }'
```

### **TEST 3: Advanced Search with Filtering**
```bash
curl -X POST http://localhost:3000/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "limit": 10,
    "filter": {
      "category": "tech"
    },
    "score_threshold": 0.5
  }'
```

### **TEST 4: Collection Management**
```bash
# Create new collection
curl -X POST http://localhost:3000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "user_profiles",
    "dimension": 128,
    "description": "User preference embeddings"
  }'

# Get collection info
curl http://localhost:3000/api/v1/collections/user_profiles
```

### **TEST 5: Vector Operations Metrics**
```bash
curl http://localhost:3000/api/v1/vectors/metrics
# Shows: insert counts, search statistics, performance metrics
```

---

## 🎯 **REAL-WORLD TESTING SCENARIOS**

### **Scenario A: Document Similarity Search**
```bash
# 1. Insert document embeddings
curl -X POST http://localhost:3000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [/* 384-dimensional vector */],
    "metadata": {
      "title": "Machine Learning Basics",
      "content": "Introduction to ML algorithms...",
      "tags": ["AI", "ML", "tutorial"],
      "author": "AI Expert"
    }
  }'

# 2. Search for similar documents
curl -X POST http://localhost:3000/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [/* query vector */],
    "limit": 5,
    "filter": {"tags": "AI"}
  }'
```

### **Scenario B: Conversation Memory**
```bash
# Store conversation context
curl -X POST http://localhost:3000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [/* conversation embedding */],
    "metadata": {
      "user_id": "user123",
      "conversation_id": "conv456",
      "prompt": "How does machine learning work?",
      "response": "Machine learning is...",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "collection": "conversations"
  }'
```

### **Scenario C: Performance Testing**
```bash
# Batch insert 100 vectors
curl -X POST http://localhost:3000/api/v1/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [/* array of 100 vectors */],
    "batch_size": 25
  }'

# Measure search performance
time curl -X POST http://localhost:3000/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [/* query */], "limit": 100}'
```

---

## 📈 **MONITORING AND VALIDATION**

### **Check System Health**
```bash
# Overall system status
curl http://localhost:3000/api/v1/system/status

# Vector-specific health
curl http://localhost:3000/api/v1/vectors/health

# Detailed metrics
curl http://localhost:3000/api/v1/vectors/metrics
```

### **Validate Performance**
```bash
# Expected metrics after testing:
# - Insert throughput: >100 vectors/sec
# - Search latency: <100ms for most queries
# - Memory usage: Efficient vector storage
# - Connection health: All connections healthy
```

---

## 🐛 **TROUBLESHOOTING**

### **Common Issues:**
1. **Qdrant not running**: Start with `docker-compose -f docker-compose.qdrant.yml up -d`
2. **Connection refused**: Check `QDRANT_URL` in `.env`
3. **Collection errors**: Verify collections are created automatically
4. **Vector dimension mismatch**: Ensure consistent vector dimensions

### **Debug Commands:**
```bash
# Check Qdrant directly
curl http://localhost:6333/collections

# Check server logs
tail -f server.log

# Verify environment variables
env | grep QDRANT
env | grep EMBEDDING
```

---

## 🎯 **SUCCESS CRITERIA**

✅ Server starts without errors
✅ Vector collections are created automatically  
✅ Vector insertion works (single and batch)
✅ Similarity search returns relevant results
✅ Collection management operations succeed
✅ Health endpoints show system is healthy
✅ Metrics show performance within expected ranges

**Your AI inference server is now a complete vector-enabled AI platform!** 🚀