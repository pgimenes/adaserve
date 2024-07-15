echo "Running autosharding for OPT model..."
echo "1. Exporting ILP solution..."
python src/main.py \
    --model opt \
    --skip-forward \
    --num_hidden_layers 1 \
    --optimizer_mip_rel_gap 95

echo "2. Loading ILP solution for forward pass..."
python src/main.py \
    --model opt \
    --preload \
    --num_hidden_layers 1 \
    --optimizer_mip_rel_gap 95