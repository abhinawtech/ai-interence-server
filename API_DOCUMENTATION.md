# AI Inference Server - API Documentation

## Base URL
```
http://localhost:8080
```

## Health Check

### GET /health
Check server health and status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-21T10:30:45Z"
}
```

## Text Generation APIs

### POST /api/v1/generate
Generate text responses with optional document context.

**Request:**
```json
{
  "prompt": "What is the remote work policy?",
  "max_tokens": 100,
  "temperature": 0.7,
  "document_content": "Optional document text for RAG"
}
```

**Response:**
```json
{
  "generated_text": "Employees may work remotely up to 3 days per week",
  "tokens_generated": 12,
  "processing_time_ms": 150
}
```

### POST /api/v1/generate/upload
Generate text responses using uploaded document for context.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Fields:
  - `file`: Document file (PDF, DOCX, TXT, MD)
  - `prompt`: Text prompt/question
  - `max_tokens`: Maximum tokens (optional, default: 100)
  - `temperature`: Temperature value (optional, default: 0.7)

**Response:**
```json
{
  "generated_text": "Based on the uploaded document, employees may work remotely up to 3 days per week",
  "tokens_generated": 15,
  "processing_time_ms": 200
}
```

## Model Management APIs

### GET /api/v1/models
List all available models.

### POST /api/v1/models
Load a new model.

### GET /api/v1/models/{model_id}
Get specific model information.

### DELETE /api/v1/models/{model_id}
Remove a model.

### POST /api/v1/models/{model_id}/health
Health check for specific model.

### GET /api/v1/models/active
Get currently active model.

### POST /api/v1/models/swap
Swap to different model.

### GET /api/v1/models/{model_id}/swap/safety
Validate model swap safety.

### POST /api/v1/models/rollback
Rollback to previous model.

### GET /api/v1/system/status
Get overall system status.

## Vector Database APIs

### POST /api/v1/vectors
Insert vector into database.

### POST /api/v1/vectors/search
Search vectors by similarity.

### GET /api/v1/vectors/stats
Get vector database statistics.

### GET /api/v1/vectors/list
List all vectors.

### GET /api/v1/vectors/{id}
Get specific vector by ID.

### DELETE /api/v1/vectors/{id}
Delete specific vector.

## Enhanced Vector APIs

### POST /api/v1/vectors/search/semantic
Advanced semantic search.

### POST /api/v1/vectors/search/advanced
Advanced search with filters.

### POST /api/v1/vectors/text
Insert text as vector.

### POST /api/v1/vectors/analyze
Analyze query semantics.

### GET /api/v1/vectors/stats/enhanced
Get enhanced statistics.

## Search APIs

### POST /api/v1/search/semantic
Semantic search with session context.

### POST /api/v1/search/contextual
Contextual search with memory.

### POST /api/v1/search/suggest
Get search suggestions.

### POST /api/v1/search/trending
Get trending search topics.

### GET /api/v1/search/analytics/{session_id}
Get search analytics for session.

## Embedding APIs

### POST /api/v1/embed
Generate embeddings for text.

### POST /api/v1/embed/batch
Batch embedding generation.

### POST /api/v1/embed/similarity
Calculate text similarity.

### GET /api/v1/embed/stats
Get embedding statistics.

### GET /api/v1/embed/{id}
Get specific embedding.

## Document Processing APIs

### POST /api/v1/documents/ingest
Ingest and process documents for RAG.

### POST /api/v1/documents/upload
Upload file for document processing.

### POST /api/v1/documents/ingest/batch
Batch document ingestion.

### GET /api/v1/documents/{id}
Get specific document.

### POST /api/v1/documents/{id}/chunk
Chunk existing document.

### POST /api/v1/documents/chunk
Chunk arbitrary content.

### GET /api/v1/documents/{id}/chunks
Get document chunks.

### POST /api/v1/documents/update
Update existing document.

### GET /api/v1/documents/{id}/versions
Get document versions.

### POST /api/v1/documents/deduplicate
Run deduplication process.

### GET /api/v1/documents/duplicates
Find duplicate documents.

### GET /api/v1/documents/stats
Get document statistics.

### GET /api/v1/documents/{id}/stats
Get specific document stats.

## Index Management APIs

### Collection Management
- `POST /collections/register` - Register new collection
- `PUT /collections/{name}/optimize` - Optimize collection
- `GET /collections/{name}/analyze` - Analyze performance
- `POST /collections/{name}/auto-optimize` - Auto-optimize
- `GET /collections/{name}/benchmark` - Benchmark configurations
- `GET /configurations` - Get all configurations

### Background Reindexing
- `POST /reindex/schedule` - Schedule reindex job
- `GET /reindex/jobs` - Get all jobs
- `GET /reindex/jobs/{job_id}` - Get job status
- `POST /reindex/jobs/{job_id}/pause` - Pause job
- `POST /reindex/jobs/{job_id}/resume` - Resume job
- `POST /reindex/jobs/{job_id}/cancel` - Cancel job
- `GET /reindex/jobs/status/{status}` - Get jobs by status
- `GET /reindex/queue/status` - Get queue status

### Performance Monitoring
- `POST /monitor/collections/{name}/metrics` - Record metric
- `GET /monitor/collections/{name}/performance` - Get performance
- `GET /monitor/collections/{name}/health` - Get health status
- `GET /monitor/collections/{name}/metrics/history` - Metric history
- `POST /monitor/collections/{name}/simulate` - Simulate metrics
- `GET /monitor/health` - Get all health statuses
- `GET /monitor/alerts` - Get active alerts
- `POST /monitor/alerts` - Create alert rule
- `POST /monitor/alerts/{id}/resolve` - Resolve alert
- `GET /monitor/alert-rules` - Get alert rules

## Error Responses

All APIs return structured error responses:

```json
{
  "error": "Invalid request",
  "message": "Missing required field 'prompt'",
  "code": 400
}
```

## Rate Limits

- 100 requests per minute per IP
- File uploads limited to 10MB
- Maximum 1000 tokens per generation request

## Authentication

Currently no authentication required. API keys will be implemented in production.

## Examples

### Generate with Document Context
```bash
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How many days can I work remotely?",
    "document_content": "Employees may work remotely up to 3 days per week",
    "max_tokens": 50
  }'
```

### Upload and Generate
```bash
curl -X POST http://localhost:8080/api/v1/generate/upload \
  -F "file=@policy.txt" \
  -F "prompt=What is the remote work policy?" \
  -F "max_tokens=100"
```

## Supported File Formats

- **Text**: .txt, .md
- **Documents**: .pdf, .docx
- **Maximum file size**: 10MB

## Response Times

- Simple generation: ~100-200ms
- With document context: ~150-300ms
- File upload processing: ~200-500ms (depending on file size)