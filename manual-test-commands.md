# 🧪 **MANUAL API TESTING COMMANDS**

## 🚀 **Quick Start Testing**

### **1. Basic Setup Verification**
```bash
# Check if server is running
curl http://localhost:3000/health

# Check if Qdrant is running
curl http://localhost:6333/collections

# Check vector database health
curl http://localhost:3000/api/v1/vectors/health
```

### **2. Before vs After Comparison**

**🔸 BEFORE Implementation:**
```bash
# Only had basic endpoints
curl http://localhost:3000/health
curl -X POST http://localhost:3000/api/v1/generate -H "Content-Type: application/json" -d '{"prompt": "Hello"}'
curl http://localhost:3000/api/v1/models
```

**🔸 AFTER Implementation:**
```bash
# Same endpoints PLUS 10+ new vector endpoints
curl http://localhost:3000/api/v1/collections
curl http://localhost:3000/api/v1/vectors/health
curl http://localhost:3000/api/v1/vectors/metrics
```

---

## 📝 **Step-by-Step Testing Guide**

### **STEP 1: Check Collections**
```bash
curl http://localhost:3000/api/v1/collections
# Expected: Shows ai_embeddings, conversations collections
```

### **STEP 2: Insert Your First Vector**
```bash
curl -X POST http://localhost:3000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "metadata": {
      "title": "My First Vector",
      "category": "test",
      "description": "Testing vector insertion"
    },
    "collection": "ai_embeddings"
  }'
```

### **STEP 3: Search for Similar Vectors**
```bash
curl -X POST http://localhost:3000/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "limit": 5,
    "score_threshold": 0.5,
    "include_vector": true
  }'
```

### **STEP 4: Batch Insert (More Realistic)**
```bash
curl -X POST http://localhost:3000/api/v1/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {
        "vector": [0.2, 0.3, 0.4, 0.5, 0.6],
        "metadata": {"title": "Document 1", "topic": "AI"}
      },
      {
        "vector": [0.3, 0.4, 0.5, 0.6, 0.7],
        "metadata": {"title": "Document 2", "topic": "ML"}
      },
      {
        "vector": [0.4, 0.5, 0.6, 0.7, 0.8],
        "metadata": {"title": "Document 3", "topic": "DL"}
      }
    ],
    "collection": "ai_embeddings"
  }'
```

### **STEP 5: Advanced Search with Filtering**
```bash
curl -X POST http://localhost:3000/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.2, 0.3, 0.4, 0.5, 0.6],
    "limit": 10,
    "filter": {
      "topic": "AI"
    },
    "score_threshold": 0.3
  }'
```

### **STEP 6: Create Custom Collection**
```bash
curl -X POST http://localhost:3000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_documents",
    "dimension": 256,
    "description": "Personal document embeddings"
  }'
```

### **STEP 7: Get Collection Info**
```bash
curl http://localhost:3000/api/v1/collections/my_documents
```

### **STEP 8: Check Metrics**
```bash
curl http://localhost:3000/api/v1/vectors/metrics
# Shows: insert counts, search performance, success rates
```

---

## 🎯 **Real-World Testing Scenarios**

### **Scenario A: Document Search System**
```bash
# 1. Insert document embeddings
curl -X POST http://localhost:3000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "metadata": {
      "title": "Introduction to Machine Learning",
      "author": "AI Expert",
      "tags": ["machine learning", "AI", "tutorial"],
      "content_type": "article",
      "word_count": 1500,
      "difficulty": "beginner"
    },
    "collection": "ai_embeddings"
  }'

# 2. Search for related documents
curl -X POST http://localhost:3000/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.95],
    "limit": 5,
    "filter": {
      "difficulty": "beginner"
    },
    "score_threshold": 0.7
  }'
```

### **Scenario B: Conversation Memory**
```bash
# Store conversation context
curl -X POST http://localhost:3000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.8],
    "metadata": {
      "user_id": "user123",
      "session_id": "sess456",
      "prompt": "Explain how neural networks work",
      "response_summary": "Explained basic neural network concepts",
      "sentiment": "positive",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "collection": "conversations"
  }'

# Find similar conversations
curl -X POST http://localhost:3000/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.8],
    "limit": 3,
    "filter": {
      "user_id": "user123"
    },
    "collection": "conversations"
  }'
```

---

## 📊 **Performance Testing**

### **Load Test: Batch Operations**
```bash
# Insert 50 vectors at once
curl -X POST http://localhost:3000/api/v1/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      /* Create array of 50 vectors with different metadata */
      {"vector": [0.1, 0.2, 0.3, 0.4, 0.5], "metadata": {"id": 1}},
      {"vector": [0.2, 0.3, 0.4, 0.5, 0.6], "metadata": {"id": 2}},
      /* ... continue for 50 vectors ... */
    ],
    "batch_size": 25
  }'
```

### **Speed Test: Search Performance**
```bash
# Time multiple searches
time curl -X POST http://localhost:3000/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "limit": 100
  }'
```

---

## 🎉 **Success Indicators**

### **✅ What You Should See:**

1. **Collections Created:**
   ```json
   {
     "collections": [
       {"name": "ai_embeddings", "vectors_count": 10},
       {"name": "conversations", "vectors_count": 5}
     ]
   }
   ```

2. **Successful Vector Insertion:**
   ```json
   {
     "id": "550e8400-e29b-41d4-a716-446655440000",
     "success": true,
     "message": "Vector inserted successfully. Inserted: 1, Failed: 0"
   }
   ```

3. **Search Results:**
   ```json
   {
     "results": [
       {
         "id": "550e8400-e29b-41d4-a716-446655440000",
         "score": 0.95,
         "metadata": {"title": "My First Vector"}
       }
     ],
     "total_found": 1,
     "search_time_ms": 25
   }
   ```

4. **Health Check:**
   ```json
   {
     "status": "healthy",
     "qdrant_connected": true,
     "collections": ["ai_embeddings", "conversations"],
     "total_vectors": 15
   }
   ```

---

## 🚨 **Troubleshooting**

### **If Vector Operations Fail:**
```bash
# Check Qdrant status
curl http://localhost:6333/health

# Check server logs for errors
tail -f server.log

# Verify collections exist
curl http://localhost:6333/collections
```

### **Common Issues:**
- **"Connection refused"**: Start Qdrant with `docker-compose -f docker-compose.qdrant.yml up -d`
- **"Collection not found"**: Collections are auto-created on first use
- **"Invalid vector dimension"**: Ensure all vectors in a collection have same dimension

---

## 🎯 **What This Proves:**

✅ **Your server now has persistent memory** (vectors are stored and searchable)
✅ **Semantic search capabilities** (find similar content by meaning)
✅ **Scalable batch operations** (handle large-scale data efficiently)
✅ **Production-ready monitoring** (health checks and metrics)
✅ **Flexible data management** (custom collections and metadata)

**You've successfully transformed a simple inference server into a comprehensive AI platform! 🚀**