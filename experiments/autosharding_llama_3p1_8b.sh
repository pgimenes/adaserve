export CHECKPOINT=/data/pedrogimenes/meta-llama/Meta-Llama-3.1-8B-Instruct/
export TENSOR_PARALLEL=4

vllm serve \
    $CHECKPOINT \
    --api_key test \
    --load_format mase \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --enforce_eager \
    --dtype float16