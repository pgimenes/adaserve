export CHECKPOINT=/home/pedrogimenes/huggingface/unsloth/Meta-Llama-3.1-8B-Instruct
export TENSOR_PARALLEL=4

vllm serve \
    $CHECKPOINT \
    --dynamic_resharding \
    --load_format mase \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --enforce_eager \
    --dtype float32