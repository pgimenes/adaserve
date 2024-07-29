export MODEL=gpt2
export CHECKPOINT=/data/huggingface/nice-gpt2-1.5b
export BATCH_SIZE=8
export SEQUENCE_LENGTH=128
export MIP_REL_GAP=98

echo "Running autosharding for ${MODEL} model..."

python src/main.py \
    --model $MODEL \
    --preload \
    --checkpoint $CHECKPOINT \
    --from_config \
    --num_hidden_layers 1 \
    --activation_function gelu \
    --batch_size $BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --optimizer_mip_rel_gap $MIP_REL_GAP