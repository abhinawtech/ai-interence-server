"""
Hugging Face Spaces interface for AI Inference Server
Provides a simple web interface using Gradio to interact with the Rust backend
"""

import gradio as gr
import requests
import json
import time
import os
import subprocess
import threading
from typing import Optional, Tuple

# Global variable to track server process
server_process = None
server_ready = False

def start_rust_server():
    """Start the Rust AI inference server in the background"""
    global server_process, server_ready
    
    try:
        print("🚀 Starting AI Inference Server...")
        
        # Start the Rust server process
        server_process = subprocess.Popen(
            ["/usr/local/bin/ai-interence-server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://localhost:7860/health", timeout=2)
                if response.status_code == 200:
                    server_ready = True
                    print("✅ AI Inference Server is ready!")
                    return
            except requests.exceptions.RequestException:
                time.sleep(2)
                print(f"⏳ Waiting for server... ({attempt + 1}/{max_attempts})")
        
        print("❌ Server failed to start within timeout")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def check_server_health() -> Tuple[bool, str]:
    """Check if the Rust server is healthy"""
    try:
        response = requests.get("http://localhost:7860/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, f"✅ Server healthy - {data.get('service', 'AI Inference Server')}"
        else:
            return False, f"❌ Server unhealthy - HTTP {response.status_code}"
    except Exception as e:
        return False, f"❌ Server connection failed: {str(e)}"

def generate_text(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    """Generate text using the AI inference server"""
    global server_ready
    
    if not server_ready:
        return "❌ Server is not ready yet. Please wait..."
    
    if not prompt.strip():
        return "❌ Please enter a prompt"
    
    try:
        # Prepare request payload
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "model": model
        }
        
        # Add temperature if supported by model
        if temperature != 1.0:
            payload["temperature"] = temperature
        
        print(f"📤 Sending request: {payload}")
        
        # Make request to Rust server
        response = requests.post(
            "http://localhost:7860/api/v1/generate",
            json=payload,
            timeout=120  # Allow up to 2 minutes for generation
        )
        
        if response.status_code == 200:
            data = response.json()
            generated_text = data.get("text", "No text generated")
            
            # Add some metadata
            metadata = []
            if "processing_time_ms" in data:
                metadata.append(f"⏱️ Time: {data['processing_time_ms']}ms")
            if "token_generated" in data:
                metadata.append(f"🎯 Tokens: {data['token_generated']}")
            
            result = generated_text
            if metadata:
                result += f"\n\n---\n📊 {' | '.join(metadata)}"
            
            return result
        else:
            error_msg = f"❌ Error {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f": {error_data['error']}"
            except:
                error_msg += f": {response.text}"
            return error_msg
            
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. The model might be loading or the prompt is too complex."
    except Exception as e:
        return f"❌ Request failed: {str(e)}"

def get_model_info() -> str:
    """Get information about available models"""
    try:
        response = requests.get("http://localhost:7860/api/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            if isinstance(models, list):
                model_info = "📋 Available Models:\n\n"
                for model in models:
                    if isinstance(model, dict):
                        name = model.get('name', 'Unknown')
                        desc = model.get('description', 'No description')
                        model_info += f"• **{name}**: {desc}\n"
                    else:
                        model_info += f"• {model}\n"
                return model_info
            else:
                return f"📋 Models: {models}"
        else:
            return "❌ Failed to fetch model information"
    except Exception as e:
        return f"❌ Error fetching models: {str(e)}"

# Start the Rust server in a separate thread
print("🔧 Starting background server...")
server_thread = threading.Thread(target=start_rust_server, daemon=True)
server_thread.start()

# Create Gradio interface
with gr.Blocks(title="🚀 AI Inference Server", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 AI Inference Server
    
    High-performance AI inference server built with Rust and Candle framework.
    Supports multiple models with optimized concurrent processing.
    """)
    
    with gr.Tab("💬 Text Generation"):
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=4
                )
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=["tinyllama", "llama-generic", "llama2", "phi3", "gemma"],
                        value="tinyllama"
                    )
                    max_tokens_slider = gr.Slider(
                        label="Max Tokens",
                        minimum=1,
                        maximum=500,
                        value=100,
                        step=1
                    )
                
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1
                )
                
                generate_btn = gr.Button("🎯 Generate", variant="primary")
                
            with gr.Column(scale=3):
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=15,
                    max_lines=20
                )
        
        generate_btn.click(
            fn=generate_text,
            inputs=[prompt_input, model_dropdown, max_tokens_slider, temperature_slider],
            outputs=output_text
        )
    
    with gr.Tab("📊 Server Status"):
        with gr.Row():
            health_btn = gr.Button("🔍 Check Health")
            models_btn = gr.Button("📋 List Models")
        
        status_output = gr.Textbox(
            label="Status",
            lines=10,
            max_lines=15
        )
        
        health_btn.click(
            fn=lambda: check_server_health()[1],
            outputs=status_output
        )
        
        models_btn.click(
            fn=get_model_info,
            outputs=status_output
        )
    
    with gr.Tab("📚 API Documentation"):
        gr.Markdown("""
        ## 🔌 API Endpoints
        
        ### Text Generation
        ```bash
        POST /api/v1/generate
        Content-Type: application/json
        
        {
            "prompt": "Your prompt here",
            "model": "tinyllama",
            "max_tokens": 100
        }
        ```
        
        ### Available Models
        - **tinyllama**: Fast 1.1B parameter model (10-14 tok/s)
        - **llama-generic**: Flexible Llama model loader
        - **llama2**: Llama-2 7B Chat model
        - **phi3**: Microsoft Phi-3 model
        - **gemma**: Google Gemma model
        
        ### Health Check
        ```bash
        GET /health
        ```
        
        ## 🚀 Features
        - Lock-free concurrent inference
        - Fast top-k sampling optimization
        - Per-request cache management
        - Multiple model support
        - Production-ready performance
        """)

# Launch the Gradio app
if __name__ == "__main__":
    print("🌐 Starting Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )