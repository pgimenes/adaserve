export MODEL=gpt2
export CHECKPOINT=/data/huggingface/nice-gpt2-1.5b
export BATCH_SIZE=8
export SEQUENCE_LENGTH=128
export MIP_REL_GAP=98

echo "Running autosharding for OPT model..."

echo "1. Exporting ILP solution..."
python src/main.py \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --num_hidden_layers 1 \
    --activation_function gelu \
    --skip-forward \
    --batch_size $BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --optimizer_mip_rel_gap $MIP_REL_GAP

echo "2. Loading ILP solution for forward pass..."
python src/main.py \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --activation_function gelu \
    --preload \
    --batch_size $BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --optimizer_mip_rel_gap $MIP_REL_GAP