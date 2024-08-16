export MODEL=gpt2
export CHECKPOINT=/data/huggingface/nice-gpt2-1.5b
export BATCH_SIZE=8
export SEQUENCE_LENGTH=128
export MIP_REL_GAP=0

echo "[EXAMPLES]: Running autosharding for ${MODEL} model..."

ada --algo alpa \
    --preload \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --batch_size $BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --optimizer_mip_rel_gap $MIP_REL_GAP