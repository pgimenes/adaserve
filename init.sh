
# Transformers nightly
echo "Installing transformers from source..."
pip install git+https://github.com/huggingface/transformers

# Mase submodule
git submodule update --init --remote
cd mase
pip install -e .
cd ..