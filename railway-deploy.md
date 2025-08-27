# Railway Deployment Guide (FREE & EASY)

## Why Railway?
- ✅ 500 hours/month FREE
- ✅ Automatic Docker builds
- ✅ Built-in monitoring
- ✅ Custom domains
- ✅ Zero config needed

## Quick Deploy Steps

### 1. Install Railway CLI
```bash
npm install -g @railway/cli
# or
curl -fsSL https://railway.app/install.sh | sh
```

### 2. Login and Initialize
```bash
railway login
railway init
```

### 3. Deploy
```bash
railway up
```

That's it! Railway will:
- Detect your Rust project
- Build using your Dockerfile
- Deploy automatically
- Give you a public URL

## Environment Variables
Set these in Railway dashboard:
```
PORT=3000
RAYON_NUM_THREADS=2
EMBEDDING_CACHE_SIZE=200
BATCH_MAX_QUEUE_SIZE=10
TOKENIZERS_PARALLELISM=false
```

## Custom Domain (Optional)
1. Go to Railway dashboard
2. Settings → Domains
3. Add your domain

## Monitoring
- Built-in metrics in Railway dashboard
- Your app's memory monitoring: `https://your-app.railway.app/api/v1/memory/stats`