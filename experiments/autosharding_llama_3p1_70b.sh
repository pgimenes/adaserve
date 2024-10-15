export CHECKPOINT=/data/huggingface/unsloth/Meta-Llama-3.1-8B-Instruct/
export TENSOR_PARALLEL=4

vllm serve \
    $CHECKPOINT \
    --api_key test \
    --load_format mase \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --enforce_eager \
    --num-parallel-layers 32 \
    --dtype float16