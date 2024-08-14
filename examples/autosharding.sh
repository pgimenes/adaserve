export MODEL=gpt2
export CHECKPOINT=/data/huggingface/nice-gpt2-1.5b
export BATCH_SIZE=8
export SEQUENCE_LENGTH=128
export MIP_REL_GAP=98

echo "[EXAMPLES]: Running autosharding for ${MODEL} model..."

python src/ada/main.py \
    --algo fully_replicated \
    --preload \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --from_config \
    --batch_size $BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH