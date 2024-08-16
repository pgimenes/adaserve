export MODEL=gpt2
export CHECKPOINT=/data/huggingface/nice-gpt2-1.5b

echo "[EXPERIMENTS]: Running sweep to extract solutions across BS/SL space for model: ${MODEL}"

python scripts/sweep.py \
    --model $MODEL \
    --checkpoint $CHECKPOINT