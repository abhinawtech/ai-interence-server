# 📮 Day 10: Postman Testing Guide for Document Processing APIs

This guide shows how to test the improved Day 10 document processing APIs using Postman, with the new clean API design that eliminates redundancy.

## 🚀 Setup

### Environment Variables
Create a Postman environment with these variables:
- `base_url`: `http://localhost:3000`
- `document_id`: (will be set dynamically from responses)

### Prerequisites
1. Start the AI inference server: `cargo run --bin ai-interence-server --release`
2. Ensure server is running on port 3000

---

## 📄 Test Collection: Document Processing Workflow

### Test 1: Health Check
**Verify server is running**

```
GET {{base_url}}/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "ai-inference-server",
  "version": "0.1.0"
}
```

---

### Test 2: Document Ingestion
**POST** `{{base_url}}/api/v1/documents/ingest`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "content": "# Machine Learning Fundamentals\n\n## Introduction\nMachine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.\n\n## Types of Machine Learning\n\n### Supervised Learning\nSupervised learning uses labeled training data to learn a mapping function from input variables (X) to output variables (Y).\n\n### Unsupervised Learning\nUnsupervised learning finds hidden patterns in data without labeled examples.\n\n### Reinforcement Learning\nReinforcement learning learns through interaction with an environment to achieve a goal.\n\n## Key Algorithms\n\n1. **Linear Regression**: Predicts continuous values\n2. **Decision Trees**: Makes decisions through a tree structure\n3. **Neural Networks**: Mimics human brain structure\n4. **Support Vector Machines**: Finds optimal decision boundaries\n\n## Applications\n- Image recognition and computer vision\n- Natural language processing\n- Recommendation systems\n- Fraud detection\n- Autonomous vehicles",
  "format": "Markdown",
  "source_path": "ml_fundamentals.md",
  "metadata": {
    "author": "AI Assistant",
    "category": "education",
    "tags": "machine-learning,AI,algorithms"
  }
}
```

**Response Handling (Tests tab):**
```javascript
// Save document_id for later tests
const response = pm.response.json();
pm.environment.set("document_id", response.document_id);

// Verify response structure
pm.test("Document ingested successfully", function () {
    pm.expect(response.document_id).to.not.be.undefined;
    pm.expect(response.sections_count).to.be.greaterThan(0);
    pm.expect(response.total_tokens).to.be.greaterThan(0);
    pm.expect(response.format).to.eql("Markdown");
});
```

**Expected Response:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "original_id": "550e8400-e29b-41d4-a716-446655440001",
  "sections_count": 8,
  "total_tokens": 324,
  "processing_time_ms": 15,
  "format": "Markdown"
}
```

---

### Test 3: Retrieve Document by ID ✨ (NEW)
**GET** `{{base_url}}/api/v1/documents/{{document_id}}`

**Tests:**
```javascript
const response = pm.response.json();

pm.test("Document retrieved successfully", function () {
    pm.expect(response.id).to.eql(pm.environment.get("document_id"));
    pm.expect(response.sections).to.be.an('array');
    pm.expect(response.total_tokens).to.be.greaterThan(0);
    pm.expect(response.format).to.eql("Markdown");
});
```

**Expected Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "original_id": "550e8400-e29b-41d4-a716-446655440001",
  "title": "Machine Learning Fundamentals",
  "sections": [
    {
      "id": "section-001",
      "section_type": "Header",
      "content": "Machine Learning Fundamentals",
      "token_count": 4,
      "metadata": {...}
    }
    // ... more sections
  ],
  "format": "Markdown",
  "total_tokens": 324,
  "processed_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "author": "AI Assistant",
    "category": "education"
  }
}
```

---

### Test 4: Chunk Existing Document ✨ (IMPROVED API)
**POST** `{{base_url}}/api/v1/documents/{{document_id}}/chunk`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "strategy": {
    "Semantic": {
      "target_size": 250,
      "boundary_types": ["Section", "Paragraph"]
    }
  },
  "config_overrides": {
    "preserve_metadata": true,
    "add_context_headers": true
  }
}
```

**Tests:**
```javascript
const response = pm.response.json();

pm.test("Document chunked successfully", function () {
    pm.expect(response.document_id).to.eql(pm.environment.get("document_id"));
    pm.expect(response.total_chunks).to.be.greaterThan(0);
    pm.expect(response.chunks).to.be.an('array');
    pm.expect(response.quality_metrics).to.not.be.undefined;
});

pm.test("Quality metrics are present", function () {
    const metrics = response.quality_metrics;
    pm.expect(metrics.boundary_preservation_score).to.be.within(0, 1);
    pm.expect(metrics.size_consistency_score).to.be.within(0, 1);
});
```

**Expected Response:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_chunks": 6,
  "total_tokens": 324,
  "average_chunk_size": 54.0,
  "processing_time_ms": 12,
  "quality_metrics": {
    "boundary_preservation_score": 0.95,
    "size_consistency_score": 0.87,
    "overlap_coverage_score": 0.0,
    "context_preservation_score": 0.92
  },
  "chunks": [
    {
      "id": "chunk-001",
      "content": "# Machine Learning Fundamentals\n\n## Introduction\nMachine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
      "token_count": 52,
      "chunk_index": 0,
      "chunk_type": "Content",
      "has_overlap": false
    }
    // ... more chunks
  ]
}
```

---

### Test 5: Chunk Arbitrary Content ✨ (NEW ENDPOINT)
**POST** `{{base_url}}/api/v1/documents/chunk`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "content": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input. The term 'deep' refers to the number of layers in the network. Traditional neural networks only contain 2-3 hidden layers, while deep networks can have hundreds of layers. Deep learning has been particularly successful in areas like computer vision, natural language processing, and speech recognition. Popular architectures include Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks (RNNs) for sequential data, and Transformer models for language understanding.",
  "strategy": {
    "SlidingWindow": {
      "size": 150,
      "overlap": 30
    }
  }
}
```

**Tests:**
```javascript
const response = pm.response.json();

pm.test("Content chunked successfully", function () {
    pm.expect(response.total_chunks).to.be.greaterThan(0);
    pm.expect(response.chunks).to.be.an('array');
});

pm.test("Sliding window overlap detected", function () {
    const hasOverlap = response.chunks.some(chunk => chunk.has_overlap === true);
    pm.expect(hasOverlap).to.be.true;
});
```

**Expected Response:**
```json
{
  "document_id": "temp-550e8400-e29b-41d4-a716-446655440002",
  "total_chunks": 3,
  "total_tokens": 147,
  "average_chunk_size": 49.0,
  "processing_time_ms": 8,
  "quality_metrics": {
    "boundary_preservation_score": 0.82,
    "size_consistency_score": 0.94,
    "overlap_coverage_score": 0.20,
    "context_preservation_score": 0.88
  },
  "chunks": [
    {
      "id": "chunk-001",
      "content": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input. The term 'deep' refers to the number of layers",
      "token_count": 50,
      "chunk_index": 0,
      "chunk_type": "Content",
      "has_overlap": false
    },
    {
      "id": "chunk-002", 
      "content": "refers to the number of layers in the network. Traditional neural networks only contain 2-3 hidden layers, while deep networks can have hundreds of layers. Deep learning has been particularly successful",
      "token_count": 48,
      "chunk_index": 1,
      "chunk_type": "Content",
      "has_overlap": true
    }
    // ... more chunks
  ]
}
```

---

### Test 6: File Upload
**POST** `{{base_url}}/api/v1/documents/upload`

**Headers:**
```
Content-Type: multipart/form-data
```

**Body (form-data):**
- Key: `file`
- Type: File
- Value: Upload a .md, .txt, or .json file

**Tests:**
```javascript
const response = pm.response.json();

pm.test("File uploaded and processed", function () {
    pm.expect(response.document_id).to.not.be.undefined;
    pm.expect(response.sections_count).to.be.greaterThan(0);
});

// Save new document ID for further testing
pm.environment.set("uploaded_document_id", response.document_id);
```

---

### Test 7: Batch Document Ingestion
**POST** `{{base_url}}/api/v1/documents/ingest/batch`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "file_paths": [
    "examples/sample1.md",
    "examples/sample2.txt", 
    "demo_config.json"
  ],
  "batch_size": 10,
  "parallel_processing": true
}
```

**Tests:**
```javascript
const response = pm.response.json();

pm.test("Batch processing completed", function () {
    pm.expect(response.total_processed).to.be.greaterThan(0);
    pm.expect(response.results).to.be.an('array');
});
```

---

### Test 8: Document Processing Statistics
**GET** `{{base_url}}/api/v1/documents/stats`

**Tests:**
```javascript
const response = pm.response.json();

pm.test("Statistics retrieved", function () {
    pm.expect(response.total_documents).to.be.at.least(1);
    pm.expect(response.total_chunks).to.be.at.least(0);
    pm.expect(response.storage_efficiency).to.be.within(0, 1);
});
```

**Expected Response:**
```json
{
  "total_documents": 3,
  "total_versions": 3,
  "total_chunks": 15,
  "unique_content_hashes": 15,
  "duplicate_groups": 0,
  "average_document_size": 245.7,
  "storage_efficiency": 1.0
}
```

---

## 🔄 Advanced Testing Scenarios

### Test 9: Different Chunking Strategies
Create multiple requests testing different strategies:

**Semantic Chunking:**
```json
{
  "strategy": {
    "Semantic": {
      "target_size": 200,
      "boundary_types": ["Section", "Paragraph", "Sentence"]
    }
  }
}
```

**Fixed Size Chunking:**
```json
{
  "strategy": {
    "FixedSize": {
      "size": 100
    }
  }
}
```

**Token-Based Chunking:**
```json
{
  "strategy": {
    "TokenBased": {
      "tokens": 150
    }
  }
}
```

### Test 10: Error Handling
Test error scenarios:

**Document Not Found:**
```
GET {{base_url}}/api/v1/documents/00000000-0000-0000-0000-000000000000
```

**Invalid Content:**
```json
{
  "content": "",
  "strategy": {"Semantic": {"target_size": 100}}
}
```

---

## 📊 Postman Collection Structure

```
📁 Day 10 Document Processing APIs
├── 📁 Setup
│   └── Health Check
├── 📁 Document Management
│   ├── Ingest Document
│   ├── Upload File
│   ├── Get Document by ID ✨
│   └── Batch Ingest
├── 📁 Improved Chunking APIs ✨
│   ├── Chunk Existing Document (by ID)
│   └── Chunk Arbitrary Content
├── 📁 Chunking Strategies
│   ├── Semantic Chunking
│   ├── Sliding Window Chunking
│   ├── Fixed Size Chunking
│   └── Token-Based Chunking
├── 📁 Statistics & Monitoring
│   └── Get Document Stats
└── 📁 Error Scenarios
    ├── Document Not Found
    └── Invalid Requests
```

---

## 🎯 Key API Improvements Demonstrated

### ✅ **Before (Redundant Design):**
```json
POST /api/v1/documents/chunk
{
  "document_id": "uuid",
  "content": "text...",  // ❌ Redundant!
  "strategy": {...}
}
```

### ✅ **After (Clean Design):**
```json
POST /api/v1/documents/{id}/chunk  // ✨ For existing documents
{
  "strategy": {...}  // Content retrieved automatically
}

POST /api/v1/documents/chunk       // ✨ For arbitrary content  
{
  "content": "text...",
  "strategy": {...}
}
```

## 🔄 Testing Workflow

1. **Setup**: Health check, environment variables
2. **Ingest**: Create documents using various methods
3. **Retrieve**: Test document retrieval by ID
4. **Chunk Existing**: Test improved API for stored documents
5. **Chunk Content**: Test new arbitrary content chunking
6. **Compare**: Verify different chunking strategies
7. **Monitor**: Check processing statistics
8. **Error Handling**: Test edge cases

This improved API design eliminates redundancy and provides a more intuitive interface for document processing operations.