# 🧪 **TESTING WITHOUT QDRANT (FALLBACK MODE)**

If you can't get Docker/Qdrant running immediately, here's how to test the core server functionality:

## **Option 1: Start Docker First**
```bash
# Check if Docker is running
docker --version

# If not running, start Docker Desktop app
# Then try: docker-compose -f docker-compose.qdrant.yml up -d
```

## **Option 2: Simple Qdrant Setup**
```bash
# Alternative simple Docker command
docker run -d --name qdrant-test -p 6333:6333 qdrant/qdrant:latest

# Verify it's running
curl http://localhost:6333/health
```

## **Option 3: Test Basic Server (Without Vector Features)**
```bash
# Start just the AI inference server
cargo run

# Test original endpoints (these will work without Qdrant)
curl http://localhost:3000/health
curl http://localhost:3000/api/v1/models
```

## **What You'll See Without Qdrant:**

✅ **These endpoints work:**
- `GET /health` - Basic health check
- `GET /api/v1/models` - Model management
- `POST /api/v1/generate` - Text generation

❌ **These endpoints will fail gracefully:**
- `GET /api/v1/vectors/health` - Will show Qdrant connection error
- `POST /api/v1/vectors` - Will return connection error
- `GET /api/v1/collections` - Will fail with clear error message

## **Expected Behavior:**
The server will start but log warnings like:
```
⚠️ Qdrant client initialization failed: Connection refused. Vector operations will be disabled.
```

This is by design - your server is fault-tolerant! 🎯