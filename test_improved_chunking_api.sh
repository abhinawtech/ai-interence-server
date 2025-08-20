#!/bin/bash

# Test script for improved chunking API design
echo "🧪 Testing Improved Chunking API Design"
echo "========================================"

BASE_URL="http://localhost:3000"

echo ""
echo "📄 Step 1: Ingest a document"
INGEST_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/documents/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "# Machine Learning Guide\n\n## Introduction\nMachine learning is a powerful subset of artificial intelligence that enables computers to learn and make decisions from data.\n\n## Key Concepts\n- **Supervised Learning**: Learning with labeled examples\n- **Unsupervised Learning**: Finding patterns in unlabeled data\n- **Deep Learning**: Neural networks with multiple layers\n\n## Applications\n1. Image recognition and computer vision\n2. Natural language processing and translation\n3. Recommendation systems for personalized content\n4. Autonomous vehicles and robotics",
    "format": "Markdown",
    "source_path": "ml_guide.md"
  }')

echo "Ingestion response:"
echo "$INGEST_RESPONSE" | jq '.'

# Extract document ID
DOC_ID=$(echo "$INGEST_RESPONSE" | jq -r '.document_id')
echo ""
echo "📋 Document ID: $DOC_ID"

echo ""
echo "🔍 Step 2: Retrieve the document by ID"
curl -s "$BASE_URL/api/v1/documents/$DOC_ID" | jq '{id, title, format, total_tokens, sections: (.sections | length)}'

echo ""
echo "🔪 Step 3: Chunk existing document by ID (improved API)"
echo "Using: POST /api/v1/documents/{id}/chunk"
CHUNK_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/documents/$DOC_ID/chunk" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": {
      "Semantic": {
        "target_size": 200,
        "boundary_types": ["Section", "Paragraph"]
      }
    }
  }')

echo "$CHUNK_RESPONSE" | jq '{document_id, total_chunks, total_tokens, quality_metrics}'

echo ""
echo "🔪 Step 4: Chunk arbitrary content (new API)"
echo "Using: POST /api/v1/documents/chunk"
CONTENT_CHUNK_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/documents/chunk" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Artificial intelligence is transforming industries. Machine learning enables computers to learn patterns from data. Deep learning uses neural networks with multiple layers to process complex information. Natural language processing helps computers understand human language.",
    "strategy": {
      "SlidingWindow": {
        "size": 100,
        "overlap": 20
      }
    }
  }')

echo "$CONTENT_CHUNK_RESPONSE" | jq '{total_chunks, total_tokens, chunks: [.chunks[] | {id, token_count, has_overlap}]}'

echo ""
echo "✅ API Improvements Demonstrated:"
echo "  • Chunk existing documents by ID: /api/v1/documents/{id}/chunk"
echo "  • Chunk arbitrary content: /api/v1/documents/chunk"
echo "  • Retrieve documents by ID: /api/v1/documents/{id}"
echo "  • No more redundant content+document_id parameters!"