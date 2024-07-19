# echo "Running row sharding for GPT2 model..."
# python src/main.py \
#     --manual \
#     --row \
#     --model gpt2 \
#     --num_hidden_layers 1

echo "Running column sharding for GPT2 model..."
python src/main.py \
    --manual \
    --column \
    --model gpt2 \
    --num_hidden_layers 1