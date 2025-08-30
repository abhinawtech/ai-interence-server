# Dockerfile for Hugging Face Spaces deployment
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Create CPU-only version of Cargo.toml for Linux builds
RUN sed -i 's/, "metal", "accelerate"//g' Cargo.toml && \
    sed -i 's/"metal", "accelerate"//g' Cargo.toml

# Build the application with optimizations for container deployment
ENV RUSTFLAGS="-C target-cpu=native -C opt-level=3"
RUN cargo build --release --bin ai-interence-server

# Runtime stage - use slim Debian image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false appuser

# Set work directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/ai-interence-server /usr/local/bin/ai-interence-server

# Copy frontend files if they exist
COPY --from=builder /app/frontend/dist ./frontend/dist 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Environment variables for Hugging Face Spaces
ENV HOST=0.0.0.0
ENV PORT=7860
ENV RUST_LOG=info
ENV RAYON_NUM_THREADS=2
ENV TOKENIZERS_PARALLELISM=false
ENV BATCH_MAX_SIZE=2
ENV BATCH_MAX_WAIT_MS=50
ENV MAX_CONCURRENT_REQUESTS=10

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose port for Hugging Face Spaces (must be 7860)
EXPOSE 7860

# Start the server
CMD ["ai-interence-server"]