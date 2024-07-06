echo "Running autosharding for toy model..."
python src/main.py

echo "Running autosharding for Bert model..."
python src/main.py --model bert --num_hidden_layers 1