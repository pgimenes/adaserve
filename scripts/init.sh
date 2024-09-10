# Setup script for adaserve.
# Installs forked vLLM with precompiled binaries and sets up models for experiments.

# Check if ADASERVE_CHECKPOINTS_PATH is set
if [ -z "$ADASERVE_CHECKPOINTS_PATH" ]; then
    echo "Error: ADASERVE_CHECKPOINTS_PATH is not set. Please set this environment variable before running the script." >&2
    exit 1
fi

export VLLM_USE_PRECOMPILED=1

# Download vLLM binaries
pip download --no-deps --only-binary :all: --dest vllm_wheel vllm
unzip vllm_wheel/* vllm/_C.abi3.so -d vllm_binaries
unzip vllm_wheel/* vllm/_core_C.abi3.so -d vllm_binaries
unzip vllm_wheel/* vllm/_moe_C.abi3.so -d vllm_binaries

# Sync submodules
git submodule update --init --remote

# Install vLLM submodule first since it overrides torch installation
cp vllm_binaries/vllm/_C.abi3.so vllm/vllm/
cp vllm_binaries/vllm/_core_C.abi3.so vllm/vllm/
cp vllm_binaries/vllm/_moe_C.abi3.so vllm/vllm/
cd vllm
pip install -e .
cd ..

# Editable install for ada
pip install -e .

# Setup models
echo "Setting up models for experiments..."
python scripts/setup_gpt2_models.py --save_path $ADASERVE_CHECKPOINTS_PATH