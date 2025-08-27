# Simple Dockerfile for Railway - AI Inference Server
FROM rust:1.89

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Create CPU-only version of Cargo.toml for Linux builds
RUN sed -i 's/, "metal", "accelerate"//g' Cargo.toml && \
    sed -i 's/"metal", "accelerate"//g' Cargo.toml

# Build the application (CPU-optimized for Linux)
ENV RUSTFLAGS="-C target-cpu=generic"
RUN cargo build --release

# Environment variables for production
ENV HOST=0.0.0.0
ENV PORT=3000
ENV RUST_LOG=info
ENV TOKENIZERS_PARALLELISM=false
ENV RAYON_NUM_THREADS=2
ENV EMBEDDING_CACHE_SIZE=200
ENV BATCH_MAX_SIZE=2
# HF_TOKEN will be set by Railway variables

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run the server
CMD ["./target/release/ai-interence-server"]