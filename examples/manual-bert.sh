echo "Running row sharding for Bert model..."
python src/main.py --manual --row --model bert --num_hidden_layers 1

echo "Running column sharding for Bert model..."
python src/main.py --manual --column --model bert --num_hidden_layers 1