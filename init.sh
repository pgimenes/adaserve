# Cleanup torch and transformers installation
pip uninstall -y torch torchvision torchaudio transformers

# Install nightly torch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Transformers nightly
echo "Installing transformers from source..."
pip install git+https://github.com/huggingface/transformers

# Mase submodule
git submodule update --init --remote
cd mase
git checkout research/alpa-light
pip install -e .
cd ..