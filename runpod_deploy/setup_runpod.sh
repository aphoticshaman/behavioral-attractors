#!/bin/bash
# RunPod setup script for Paper B experiment
# Run this after uploading the zip to RunPod

set -e

echo "=== Paper B RunPod Setup ==="

# Install dependencies
pip install -q vllm aiohttp aiofiles pyyaml numpy pandas scipy tqdm

# Model mappings (Ollama name -> HuggingFace name)
declare -A MODELS
MODELS["phi3:3.8b"]="microsoft/Phi-3-mini-4k-instruct"
MODELS["mistral:7b"]="mistralai/Mistral-7B-Instruct-v0.2"
MODELS["llama3.2:3b"]="meta-llama/Llama-3.2-3B-Instruct"
MODELS["qwen2.5:7b"]="Qwen/Qwen2.5-7B-Instruct"

# Detect GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected $GPU_COUNT GPUs"

# Start vLLM server
# For single GPU: serve all models sequentially
# For multi-GPU: can serve one model per GPU

if [ "$GPU_COUNT" -ge 4 ]; then
    echo "Multi-GPU mode: Starting 4 vLLM instances"

    # Start each model on a different GPU
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model microsoft/Phi-3-mini-4k-instruct \
        --port 8000 --gpu-memory-utilization 0.9 &

    CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --port 8001 --gpu-memory-utilization 0.9 &

    CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --port 8002 --gpu-memory-utilization 0.9 &

    CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --port 8003 --gpu-memory-utilization 0.9 &

    echo "Waiting for servers to start..."
    sleep 60

else
    echo "Single/Few GPU mode: Running models sequentially"
    # Will use run_sequential.py instead
fi

echo "Setup complete!"
