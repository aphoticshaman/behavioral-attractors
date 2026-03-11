#!/bin/bash
# Run all 4 models - requires 4+ GPUs or run sequentially
# Usage: ./run_all_models.sh [--smoke]

set -e

SMOKE_FLAG=""
MAX_CONCURRENT=32
if [ "$1" == "--smoke" ]; then
    SMOKE_FLAG="--smoke"
    MAX_CONCURRENT=8
    echo "=== SMOKE TEST MODE ==="
fi

echo "=== Paper B Full Multi-Model Run ==="
echo "Started at: $(date)"

pip install -q vllm aiohttp aiofiles pyyaml numpy pandas scipy matplotlib seaborn tqdm

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPUs: $GPU_COUNT"

cd /workspace/paper_b

# Models (HuggingFace IDs)
MODELS=(
    "microsoft/Phi-3-mini-4k-instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

# Clear previous results
rm -f data/outputs.jsonl

if [ "$GPU_COUNT" -ge 4 ]; then
    echo "=== Parallel Mode: 4 GPUs ==="

    # Start all vLLM servers
    for i in 0 1 2 3; do
        PORT=$((8000 + i))
        echo "Starting ${MODELS[$i]} on GPU $i port $PORT"
        CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
            --model "${MODELS[$i]}" \
            --port $PORT \
            --gpu-memory-utilization 0.9 \
            --max-model-len 4096 \
            --trust-remote-code &
    done

    echo "Waiting for all servers..."
    sleep 90

    # Run experiments in parallel across all models
    for i in 0 1 2 3; do
        PORT=$((8000 + i))
        echo "Running experiments for ${MODELS[$i]}"
        python runpod_deploy/parallel_runner.py \
            --config configs/experiment.yaml \
            --vllm-url "http://localhost:$PORT" \
            --models "${MODELS[$i]}" \
            --max-concurrent $MAX_CONCURRENT \
            $SMOKE_FLAG &
    done

    wait
    echo "All model runs complete"

    # Merge results
    cat data/outputs_*.jsonl > data/outputs.jsonl 2>/dev/null || true

else
    echo "=== Sequential Mode: $GPU_COUNT GPU(s) ==="

    for MODEL in "${MODELS[@]}"; do
        echo "=== Running $MODEL ==="

        # Start vLLM
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --port 8000 \
            --gpu-memory-utilization 0.9 \
            --max-model-len 4096 \
            --trust-remote-code &
        VLLM_PID=$!

        sleep 60  # Wait for model load

        # Run experiment
        python runpod_deploy/parallel_runner.py \
            --config configs/experiment.yaml \
            --vllm-url http://localhost:8000 \
            --models "$MODEL" \
            --max-concurrent $MAX_CONCURRENT \
            $SMOKE_FLAG

        kill $VLLM_PID
        sleep 5
    done
fi

# Analysis
echo "Running analysis..."
python analysis/analyze.py --input data/outputs.jsonl --output analysis

echo "=== Complete ==="
echo "Finished at: $(date)"
zip -r results.zip data/ analysis/
ls -lh results.zip
