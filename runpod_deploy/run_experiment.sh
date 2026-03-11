#!/bin/bash
# One-click experiment runner for RunPod
# Usage: ./run_experiment.sh [--smoke]

set -e

SMOKE_FLAG=""
if [ "$1" == "--smoke" ]; then
    SMOKE_FLAG="--smoke"
    echo "=== SMOKE TEST MODE ==="
fi

echo "=== Paper B Experiment Runner ==="
echo "Started at: $(date)"

# Install deps
echo "Installing dependencies..."
pip install -q vllm aiohttp aiofiles pyyaml numpy pandas scipy matplotlib seaborn tqdm

# Detect GPUs
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPUs detected: $GPU_COUNT"

# HuggingFace token (set this in RunPod secrets or env)
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Some models may fail to download."
fi

cd /workspace/paper_b

# Model to use (pick best available for GPU memory)
# H200 has 80GB - can run 7B models easily
MODEL="mistralai/Mistral-7B-Instruct-v0.2"

echo "Starting vLLM server with $MODEL..."
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server
echo "Waiting for vLLM to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready!"
        break
    fi
    sleep 2
done

# Run experiment
echo "Starting experiment..."
python runpod_deploy/parallel_runner.py \
    --config configs/experiment.yaml \
    --vllm-url http://localhost:8000 \
    --models "$MODEL" \
    --max-concurrent 16 \
    $SMOKE_FLAG

# Run analysis
echo "Running analysis..."
python analysis/analyze.py --input data/outputs.jsonl --output analysis

# Cleanup
kill $VLLM_PID 2>/dev/null || true

echo "=== Experiment Complete ==="
echo "Finished at: $(date)"
echo "Results in: data/outputs.jsonl"
echo "Figures in: analysis/figures/"

# Zip results for download
zip -r results.zip data/ analysis/
echo "Download: results.zip"
