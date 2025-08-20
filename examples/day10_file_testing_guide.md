# 📁 Day 10: File-Based Document Testing Guide

This guide shows how to test Day 10's document processing APIs with **actual files** instead of just curl commands with JSON payloads.

## 🚀 Quick Start with File Upload

### Step 1: Start the Server
```bash
cd /Users/abhinawkumar/Code/ai-cloud-native/ai-interence-server
RUST_LOG=info cargo run --bin ai-interence-server
```

### Step 2: Create Demo Documents
```bash
# Create demo documents for testing
cargo run --bin upload_test_document --demo
```

This creates:
- `demo_ml_guide.md` - Markdown document about machine learning
- `demo_config.json` - JSON configuration file
- `demo_data.csv` - CSV data file

### Step 3: Upload Documents
```bash
# Upload the markdown file
cargo run --bin upload_test_document demo_ml_guide.md

# Upload the JSON config
cargo run --bin upload_test_document demo_config.json

# Upload the CSV data
cargo run --bin upload_test_document demo_data.csv
```

## 📄 Testing Different Document Types

### Test 1: Markdown Documents
Create a technical document:
```bash
cat > technical_guide.md << 'EOF'
# Transformer Architecture Deep Dive

## Introduction
Transformers have revolutionized natural language processing through their attention mechanism.

## Self-Attention Mechanism
The core innovation of transformers is the self-attention mechanism that allows the model to weigh the importance of different words in a sequence.

### Mathematical Foundation
The attention function can be described as:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

## Multi-Head Attention
Multi-head attention runs several attention mechanisms in parallel:

1. **Parallel Processing**: Multiple attention heads capture different types of relationships
2. **Diverse Representations**: Each head learns different aspects of the input
3. **Improved Performance**: Combination of heads provides richer understanding

## Code Implementation
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
    def forward(self, query, key, value):
        # Split into multiple heads
        # Apply attention
        # Concatenate results
        return output
```

## Applications
Transformers are now used in:
- Language models (GPT, BERT)
- Computer vision (Vision Transformer)
- Protein folding (AlphaFold)
- Code generation (Codex)

## Conclusion
The transformer architecture continues to drive advances in AI across multiple domains.
EOF

# Upload and test
cargo run --bin upload_test_document technical_guide.md
```

### Test 2: JSON Configuration Files
Create a complex configuration:
```bash
cat > ai_system_config.json << 'EOF'
{
  "system_config": {
    "name": "AI Document Processing System",
    "version": "2.0.1",
    "modules": {
      "document_ingestion": {
        "enabled": true,
        "formats": ["markdown", "json", "csv", "txt"],
        "batch_size": 100,
        "parallel_processing": true
      },
      "intelligent_chunking": {
        "enabled": true,
        "strategies": [
          {
            "name": "semantic",
            "target_size": 500,
            "boundary_types": ["paragraph", "section", "sentence"]
          },
          {
            "name": "sliding_window", 
            "size": 300,
            "overlap": 75
          }
        ]
      },
      "deduplication": {
        "enabled": true,
        "similarity_threshold": 0.90,
        "auto_run_interval": "24h"
      }
    },
    "performance": {
      "max_concurrent_uploads": 50,
      "chunking_quality_threshold": 0.85,
      "storage_optimization": true
    },
    "monitoring": {
      "metrics": ["throughput", "accuracy", "storage_efficiency"],
      "alerts": {
        "low_quality_chunks": "< 0.8",
        "high_storage_usage": "> 90%",
        "processing_errors": "> 5%"
      }
    }
  }
}
EOF

cargo run --bin upload_test_document ai_system_config.json
```

### Test 3: CSV Data Files
Create structured data:
```bash
cat > research_data.csv << 'EOF'
study_id,model_type,accuracy,inference_time_ms,memory_mb,dataset_size,architecture
S001,BERT-Base,0.92,45,512,100000,transformer
S002,GPT-3.5,0.89,120,1024,50000,autoregressive
S003,T5-Large,0.94,85,768,150000,encoder-decoder
S004,RoBERTa,0.91,38,512,120000,transformer
S005,DeBERTa,0.93,52,640,80000,transformer
S006,ALBERT,0.88,35,256,90000,transformer
S007,XLNet,0.90,65,512,110000,autoregressive
S008,DistilBERT,0.86,25,256,100000,transformer
S009,ELECTRA,0.92,42,512,95000,discriminative
S010,BigBird,0.89,95,1024,200000,sparse-attention
EOF

cargo run --bin upload_test_document research_data.csv
```

### Test 4: Plain Text Documents
Create documentation:
```bash
cat > user_manual.txt << 'EOF'
AI Document Processing System User Manual

Welcome to the AI Document Processing System, a comprehensive platform for intelligent document analysis and processing.

GETTING STARTED

Installation Requirements
- Operating System: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- Memory: Minimum 8GB RAM, recommended 16GB
- Storage: 10GB available space for installation
- Network: Stable internet connection for cloud features

First-Time Setup
1. Download the installer from the official website
2. Run the installation wizard with administrator privileges
3. Complete the initial configuration by providing your API keys
4. Test the installation with the built-in sample documents
5. Configure your preferred document processing settings

CORE FEATURES

Document Ingestion
The system supports multiple document formats including Markdown, JSON, CSV, and plain text. Documents can be uploaded individually or in batches for bulk processing.

Intelligent Chunking
Advanced algorithms automatically segment documents into logical chunks while preserving semantic meaning and context relationships.

Version Management
Track document changes over time with automatic version control and change detection capabilities.

Deduplication
Automatically identify and remove duplicate content to optimize storage and improve search performance.

BEST PRACTICES

Document Preparation
- Ensure documents are properly formatted before upload
- Use clear headings and section markers in Markdown
- Validate JSON structure before ingestion
- Include relevant metadata when possible

Performance Optimization
- Use batch uploads for multiple documents
- Enable parallel processing for large datasets
- Regular deduplication to maintain system efficiency
- Monitor quality metrics to ensure optimal chunking

TROUBLESHOOTING

Common Issues
If documents fail to upload, check file format and size limits. For chunking quality issues, adjust the target chunk size or boundary detection settings.

Support Resources
Visit our documentation portal for detailed guides, API references, and community forums for additional assistance.
EOF

cargo run --bin upload_test_document user_manual.txt
```

## 🧠 Testing Chunking with Real Files

After uploading documents, test different chunking strategies:

### Test Semantic Chunking
```bash
curl -X POST http://localhost:3000/api/v1/documents/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "'$(cat technical_guide.md | sed 's/"/\\"/g' | tr '\n' ' ')'",
    "strategy": {
      "Semantic": {
        "target_size": 400,
        "boundary_types": ["Section", "Paragraph", "CodeBlock"]
      }
    }
  }' | jq
```

### Test Sliding Window Chunking
```bash
curl -X POST http://localhost:3000/api/v1/documents/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "550e8400-e29b-41d4-a716-446655440001", 
    "content": "'$(cat user_manual.txt | sed 's/"/\\"/g' | tr '\n' ' ')'",
    "strategy": {
      "SlidingWindow": {
        "size": 200,
        "overlap": 50
      }
    }
  }' | jq
```

## 📁 Batch Upload Testing

### Test Multiple File Upload
Create a test directory with multiple files:
```bash
mkdir test_docs
cp demo_*.* test_docs/
cp technical_guide.md test_docs/
cp ai_system_config.json test_docs/

# Test batch ingestion
curl -X POST http://localhost:3000/api/v1/documents/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "test_docs/demo_ml_guide.md",
      "test_docs/technical_guide.md", 
      "test_docs/ai_system_config.json",
      "test_docs/demo_data.csv"
    ],
    "batch_size": 5,
    "parallel_processing": true
  }' | jq
```

## 🔄 Testing Document Updates and Versions

### Test Document Evolution
```bash
# Create initial version
echo "# API Documentation v1.0" > api_docs.md
echo "Basic API endpoints and usage." >> api_docs.md

cargo run --bin upload_test_document api_docs.md

# Update the document
echo "# API Documentation v2.0" > api_docs.md
echo "Enhanced API with new features and improved performance." >> api_docs.md
echo "## New Features" >> api_docs.md  
echo "- Document processing pipeline" >> api_docs.md
echo "- Intelligent chunking" >> api_docs.md
echo "- Version management" >> api_docs.md

cargo run --bin upload_test_document api_docs.md

# Check version history
curl http://localhost:3000/api/v1/documents/{document-id}/versions | jq
```

## 📊 Integration with Generate API

### Test Document-Aware Conversations
```bash
# First upload a knowledge base
cat > ml_knowledge.md << 'EOF'
# Machine Learning Fundamentals

## Supervised Learning
Supervised learning uses labeled training data to learn a mapping from inputs to outputs.

Examples:
- Classification: Predicting categories (spam/not spam)
- Regression: Predicting continuous values (house prices)

## Unsupervised Learning  
Unsupervised learning finds patterns in data without labeled examples.

Examples:
- Clustering: Grouping similar data points
- Dimensionality reduction: Simplifying data representation

## Deep Learning
Deep learning uses neural networks with multiple layers to learn complex patterns.

Key architectures:
- Convolutional Neural Networks (CNNs) for image processing
- Recurrent Neural Networks (RNNs) for sequence data  
- Transformers for natural language processing
EOF

cargo run --bin upload_test_document ml_knowledge.md

# Now test generate API with document context
curl -X POST http://localhost:3000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the difference between supervised and unsupervised learning",
    "max_tokens": 200,
    "use_memory": true,
    "session_id": "ml-education"
  }' | jq '.text'

# Follow-up question using document context
curl -X POST http://localhost:3000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Give me examples of deep learning architectures mentioned in the knowledge base",
    "max_tokens": 150, 
    "use_memory": true,
    "session_id": "ml-education"
  }' | jq '.text'
```

## 🔍 Testing Deduplication

### Create Duplicate Content for Testing
```bash
# Create similar documents
echo "Machine learning is a subset of AI." > doc1.txt
echo "Machine learning is a subset of artificial intelligence." > doc2.txt
echo "ML is a subset of AI technology." > doc3.txt

cargo run --bin upload_test_document doc1.txt
cargo run --bin upload_test_document doc2.txt  
cargo run --bin upload_test_document doc3.txt

# Test deduplication
curl -X POST http://localhost:3000/api/v1/documents/deduplicate \
  -H "Content-Type: application/json" \
  -d '{
    "similarity_threshold": 0.80
  }' | jq

# Check for duplicate candidates
curl http://localhost:3000/api/v1/documents/duplicates | jq
```

## 📈 Monitoring and Statistics

### View Processing Statistics
```bash
# Overall statistics
curl http://localhost:3000/api/v1/documents/stats | jq

# Document-specific stats
curl http://localhost:3000/api/v1/documents/{document-id}/stats | jq
```

## 🎯 Real-World Testing Scenarios

### Scenario 1: Technical Documentation
1. Upload README files, API docs, tutorials
2. Test chunking preserves code blocks and sections
3. Verify version tracking as docs evolve
4. Test search and retrieval for technical queries

### Scenario 2: Research Paper Processing
1. Upload academic papers in Markdown format
2. Test semantic chunking on abstract, methodology, results
3. Test deduplication across similar papers
4. Generate summaries and insights

### Scenario 3: Configuration Management
1. Upload JSON/YAML config files
2. Test structured parsing and chunking
3. Track configuration changes over time
4. Generate documentation from configs

## 🏆 Success Metrics

After testing, you should see:
- ✅ **Documents processed**: Various formats ingested successfully
- ✅ **Quality chunking**: High boundary preservation scores (>0.85)
- ✅ **Version tracking**: Document evolution captured accurately  
- ✅ **Deduplication**: Storage efficiency improvements
- ✅ **Generate integration**: Document-aware responses
- ✅ **Performance**: Consistent processing speeds

This file-based testing approach demonstrates Day 10's real-world capabilities for processing actual documents rather than just API testing with synthetic data.