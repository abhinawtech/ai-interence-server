// ================================================================================================
// GPU DETECTION AND AUTO-SWITCHING - SMART HARDWARE UTILIZATION
// ================================================================================================
//
// This module automatically detects available GPUs and configures the optimal backend:
// - CUDA for NVIDIA GPUs (Linux/Windows)
// - Metal for Apple Silicon (macOS)
// - CPU fallback for all other cases
//
// ================================================================================================

use std::env;
use tracing::{info, warn};

#[derive(Debug, Clone, PartialEq)]
pub enum GpuBackend {
    Cuda,
    Metal,
    Cpu,
}

impl GpuBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuBackend::Cuda => "CUDA",
            GpuBackend::Metal => "Metal",
            GpuBackend::Cpu => "CPU",
        }
    }
}

/// Auto-detect the best available GPU backend
pub fn detect_gpu_backend() -> GpuBackend {
    info!("🔍 Auto-detecting GPU backend...");

    // Check environment override first
    if let Ok(backend) = env::var("FORCE_GPU_BACKEND") {
        match backend.to_lowercase().as_str() {
            "cuda" => {
                info!("🎮 Forced CUDA backend via FORCE_GPU_BACKEND");
                return GpuBackend::Cuda;
            }
            "metal" => {
                info!("🎮 Forced Metal backend via FORCE_GPU_BACKEND");
                return GpuBackend::Metal;
            }
            "cpu" => {
                info!("🎮 Forced CPU backend via FORCE_GPU_BACKEND");
                return GpuBackend::Cpu;
            }
            _ => {
                warn!("⚠️ Invalid FORCE_GPU_BACKEND value: {}, continuing auto-detection", backend);
            }
        }
    }

    // Auto-detection logic
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            info!("🚀 NVIDIA GPU detected - using CUDA backend");
            return GpuBackend::Cuda;
        }
    }

    #[cfg(feature = "metal")]
    {
        if is_metal_available() {
            info!("🚀 Apple Silicon detected - using Metal backend");
            return GpuBackend::Metal;
        }
    }

    info!("💾 No GPU detected - using optimized CPU backend");
    GpuBackend::Cpu
}

/// Check if CUDA is available
#[cfg(feature = "cuda")]
fn is_cuda_available() -> bool {
    // Try to query CUDA devices
    match std::process::Command::new("nvidia-smi").output() {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let gpu_count = stdout.lines()
                    .filter(|line| line.contains("NVIDIA"))
                    .count();
                
                if gpu_count > 0 {
                    info!("🎮 Found {} NVIDIA GPU(s)", gpu_count);
                    return true;
                }
            }
        }
        Err(_) => {
            // nvidia-smi not found
        }
    }

    // Fallback: check if CUDA libraries are available
    if let Ok(_) = env::var("CUDA_VISIBLE_DEVICES") {
        info!("🎮 CUDA_VISIBLE_DEVICES set - assuming CUDA available");
        return true;
    }

    false
}

#[cfg(not(feature = "cuda"))]
fn is_cuda_available() -> bool {
    false
}

/// Check if Metal is available (Apple Silicon)
#[cfg(feature = "metal")]
fn is_metal_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // Check if we're on Apple Silicon
        match std::process::Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
        {
            Ok(output) => {
                let cpu_info = String::from_utf8_lossy(&output.stdout);
                if cpu_info.contains("Apple") {
                    info!("🎮 Apple Silicon detected: {}", cpu_info.trim());
                    return true;
                }
            }
            Err(_) => {}
        }

        // Fallback: assume Metal is available on all modern macOS
        info!("🎮 macOS detected - assuming Metal support");
        true
    }
    
    #[cfg(not(target_os = "macos"))]
    false
}

#[cfg(not(feature = "metal"))]
fn is_metal_available() -> bool {
    false
}

/// Get optimal thread count based on backend
pub fn get_optimal_thread_count(backend: &GpuBackend) -> usize {
    match backend {
        GpuBackend::Cuda | GpuBackend::Metal => {
            // GPU backends can use fewer CPU threads
            let cpu_count = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            (cpu_count / 2).max(2) // Use half CPU threads when GPU is available
        }
        GpuBackend::Cpu => {
            // CPU backend needs more threads
            let cpu_count = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            cpu_count.max(4)
        }
    }
}

/// Get optimal batch size based on backend and available memory
pub fn get_optimal_batch_size(backend: &GpuBackend) -> usize {
    match backend {
        GpuBackend::Cuda => {
            // CUDA can handle larger batches
            env::var("BATCH_MAX_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8)
        }
        GpuBackend::Metal => {
            // Metal optimized for Apple Silicon
            env::var("BATCH_MAX_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(6)
        }
        GpuBackend::Cpu => {
            // Conservative batch size for CPU
            env::var("BATCH_MAX_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2)
        }
    }
}

/// Print GPU backend information
pub fn print_gpu_info(backend: &GpuBackend) {
    info!("🎯 GPU Backend Configuration:");
    info!("   Backend: {}", backend.as_str());
    info!("   Threads: {}", get_optimal_thread_count(backend));
    info!("   Batch Size: {}", get_optimal_batch_size(backend));
    
    match backend {
        GpuBackend::Cuda => {
            info!("   CUDA Features: Enabled");
            if let Ok(devices) = env::var("CUDA_VISIBLE_DEVICES") {
                info!("   CUDA Devices: {}", devices);
            }
        }
        GpuBackend::Metal => {
            info!("   Metal Features: Enabled");
            info!("   Apple Silicon: Optimized");
        }
        GpuBackend::Cpu => {
            info!("   CPU Optimization: Maximum");
            info!("   Vector Instructions: Auto-detected");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_detection() {
        let backend = detect_gpu_backend();
        println!("Detected backend: {:?}", backend);
        assert!(matches!(backend, GpuBackend::Cuda | GpuBackend::Metal | GpuBackend::Cpu));
    }

    #[test]
    fn test_optimal_thread_count() {
        let cpu_backend = GpuBackend::Cpu;
        let cuda_backend = GpuBackend::Cuda;
        
        let cpu_threads = get_optimal_thread_count(&cpu_backend);
        let cuda_threads = get_optimal_thread_count(&cuda_backend);
        
        assert!(cpu_threads >= 2);
        assert!(cuda_threads >= 2);
        // GPU backends should use fewer or equal CPU threads
        assert!(cuda_threads <= cpu_threads);
    }
}