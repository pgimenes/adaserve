echo "Running row sharding for OPT model..."
python src/main.py --manual --row --model opt --num_hidden_layers 1

echo "Running column sharding for OPT model..."
python src/main.py --manual --column --model opt --num_hidden_layers 1