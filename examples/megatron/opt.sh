export CHECKPOINT=facebook/opt-1.3b
export BATCH_SIZE=128
export SEQUENCE_LENGTH=128

echo "Running autosharding for OPT model..."

echo "1. Sharding model with Megatron-LM algorithm..."
python src/main.py \
    --model opt \
    --checkpoint $CHECKPOINT \
    --num_hidden_layers 1 \
    --skip-forward \
    --algo megatron \
    --batch_size $BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --optimizer_mip_rel_gap 98

# echo "2. Loading sharding configuration for forward pass..."
# python src/main.py \
#     --model opt \
#     --checkpoint $CHECKPOINT \
#     --preload \
#     --algo megatron \
#     --batch_size $BATCH_SIZE \
#     --sequence_length $SEQUENCE_LENGTH \
#     --optimizer_mip_rel_gap 98