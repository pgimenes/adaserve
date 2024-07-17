export CHECKPOINT=facebook/opt-1.3b
export BATCH_SIZE=128
export SEQUENCE_LENGTH=128

echo "Running autosharding for OPT model..."

echo "1. Exporting ILP solution..."
python src/main.py \
    --model opt \
    --checkpoint $CHECKPOINT \
    --skip-forward \
    --batch_size $BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --optimizer_mip_rel_gap 98

echo "2. Loading ILP solution for forward pass..."
python src/main.py \
    --model opt \
    --checkpoint $CHECKPOINT \
    --preload \
    --batch_size $BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --optimizer_mip_rel_gap 98