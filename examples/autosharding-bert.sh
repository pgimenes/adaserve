echo "Running autosharding for Bert model..."
python src/main.py \
    --model bert \
    --num_hidden_layers 1 \
    --optimizer_mip_rel_gap 99