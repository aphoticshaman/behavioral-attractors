#!/bin/bash
# Optimized for 7x H200 SXM
# Expected runtime: ~30-45 minutes for full run

set -e

echo "=============================================="
echo "Paper B - 7x H200 Optimized Run"
echo "Started: $(date)"
echo "=============================================="

# Install deps (vllm template has most, but ensure all)
pip install -q aiohttp aiofiles pyyaml numpy pandas scipy matplotlib seaborn tqdm

cd /workspace/paper_b

# Models - run on GPUs 0-3, leave 4-6 for batching headroom
MODELS=(
    "microsoft/Phi-3-mini-4k-instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

PORTS=(8000 8001 8002 8003)

echo "Starting 4 vLLM servers..."

for i in 0 1 2 3; do
    echo "GPU $i: ${MODELS[$i]} on port ${PORTS[$i]}"
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --model "${MODELS[$i]}" \
        --port ${PORTS[$i]} \
        --gpu-memory-utilization 0.95 \
        --max-model-len 4096 \
        --disable-log-requests \
        --trust-remote-code \
        > /tmp/vllm_$i.log 2>&1 &
done

echo "Waiting for servers to initialize (90s)..."
sleep 90

# Verify all servers are up
echo "Checking server health..."
for i in 0 1 2 3; do
    PORT=${PORTS[$i]}
    if curl -s "http://localhost:$PORT/health" > /dev/null; then
        echo "  Port $PORT: OK"
    else
        echo "  Port $PORT: FAILED - check /tmp/vllm_$i.log"
        cat /tmp/vllm_$i.log | tail -20
    fi
done

# Clear previous outputs
rm -f data/outputs*.jsonl

echo ""
echo "=============================================="
echo "Running experiments in parallel..."
echo "=============================================="

# Run all 4 models in parallel with high concurrency
for i in 0 1 2 3; do
    PORT=${PORTS[$i]}
    MODEL="${MODELS[$i]}"
    OUTFILE="data/outputs_$i.jsonl"

    echo "Launching: $MODEL"
    python runpod_deploy/parallel_runner.py \
        --config configs/experiment.yaml \
        --vllm-url "http://localhost:$PORT" \
        --models "$MODEL" \
        --max-concurrent 64 \
        > "/tmp/run_$i.log" 2>&1 &
done

echo "All 4 model runs launched. Monitoring..."

# Wait for all to complete
wait

echo ""
echo "=============================================="
echo "Merging results..."
echo "=============================================="

# Merge all outputs
cat data/outputs*.jsonl > data/outputs_merged.jsonl 2>/dev/null || true
mv data/outputs_merged.jsonl data/outputs.jsonl

TRIAL_COUNT=$(wc -l < data/outputs.jsonl)
echo "Total trials: $TRIAL_COUNT"

echo ""
echo "=============================================="
echo "Running analysis..."
echo "=============================================="

python analysis/analyze.py --input data/outputs.jsonl --output analysis

echo ""
echo "=============================================="
echo "COMPLETE"
echo "=============================================="
echo "Finished: $(date)"
echo ""
echo "Results:"
ls -lh data/outputs.jsonl
ls -lh analysis/figures/ 2>/dev/null || echo "No figures generated"
echo ""

# Package for download
zip -r results.zip data/ analysis/
echo "Download: results.zip ($(ls -lh results.zip | awk '{print $5}'))"

# Kill vLLM servers
pkill -f "vllm.entrypoints" || true

echo ""
echo "Pod can be stopped now."
