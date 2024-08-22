# Setup script for adaserve.
# Need to install in this order: vLLM -> torch -> transformers -> mase

# Mase needs nightly torch/transformers, but these packages are overwritten by
# default vLLM installation.

# We install forked vLLM with pre-compiled binaries from the main package.

export VLLM_USE_PRECOMPILED=1

# Download vLLM binaries
pip download --no-deps --only-binary :all: --dest . vllm
unzip vllm-0.5.4-cp38-abi3-manylinux1_x86_64.whl vllm/_C.abi3.so -d vllm_binaries
unzip vllm-0.5.4-cp38-abi3-manylinux1_x86_64.whl vllm/_core_C.abi3.so -d vllm_binaries
unzip vllm-0.5.4-cp38-abi3-manylinux1_x86_64.whl vllm/_moe_C.abi3.so -d vllm_binaries

# Sync submodules
git submodule update --init --remote

# Install vLLM submodule first since it overrides torch installation
cp vllm_binaries/vllm/_C.abi3.so vllm/vllm/
cp vllm_binaries/vllm/_core_C.abi3.so vllm/vllm/
cp vllm_binaries/vllm/_moe_C.abi3.so vllm/vllm/
cd vllm
pip install -e .
cd ..

# Cleanup torch and transformers installations and install nightly
pip uninstall -y torch torchvision torchaudio transformers

echo "Installing torch from source..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

echo "Installing transformers from source..."
pip install git+https://github.com/huggingface/transformers

# Mase submodule
cd mase
git checkout research/alpa-light
pip install -e .
cd ..

# Setup models
echo "Setting up models for experiments..."
python experiments/setup_gpt2_models.py --save_path $ADASERVE_CHECKPOINTS_PATH