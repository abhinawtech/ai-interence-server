# Render Deployment Guide (FREE)

## Why Render?
- ✅ FREE tier with 750 hours/month
- ✅ Auto-deploys from GitHub
- ✅ Built-in SSL
- ✅ Easy scaling

## Deploy Steps

### 1. Connect GitHub
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New Web Service"
4. Connect your repository

### 2. Configure Service
```yaml
Name: ai-inference-server
Environment: Docker
Branch: main
Docker Command: (leave blank - uses Dockerfile)
```

### 3. Environment Variables
Add these in Render dashboard:
```
PORT=10000
RAYON_NUM_THREADS=1
EMBEDDING_CACHE_SIZE=100
BATCH_MAX_QUEUE_SIZE=5
TOKENIZERS_PARALLELISM=false
```

### 4. Deploy
Click "Create Web Service" - automatic deployment starts!

## Free Tier Limits
- 750 hours/month
- Sleeps after 15 minutes of inactivity
- 512MB RAM
- Shared CPU

## Custom Domain
Available on paid plans ($7/month)