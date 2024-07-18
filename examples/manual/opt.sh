echo "Running row sharding for OPT model..."
python src/main.py \
    --manual \
    --row \
    --model opt \
    --num_hidden_layers 1 \
    --_attn_implementation flash_attention_2

echo "Running column sharding for OPT model..."
python src/main.py \
    --manual \
    --column \
    --model opt \
    --num_hidden_layers 1