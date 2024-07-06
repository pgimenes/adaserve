echo "Running row sharding for toy model..."
python src/main.py --manual --row

echo "Running column sharding for toy model..."
python src/main.py --manual --column

echo "Running row sharding for Bert model..."
python src/main.py --manual --row --model bert --num_hidden_layers 1

echo "Running column sharding for Bert model..."
python src/main.py --manual --column --model bert --num_hidden_layers 1