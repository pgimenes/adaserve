# Run sweep of batch size and sequence length for single-layer bert model
python src/main.py \
    --sweep \
    --model bert \
    --num_hidden_layers 1 \
    --optimizer_mip_rel_gap 95 \
    --sweep-grid-size 5 \
    --sweep-max-threads 16