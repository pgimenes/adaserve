export CHECKPOINT=/data/pedrogimenes/meta-llama/Meta-Llama-3.1-70B-Instruct/
export TENSOR_PARALLEL=8

vllm serve \
    $CHECKPOINT \
    --api_key test \
    --load_format mase \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --enforce_eager \
    --num-parallel-layers 0 \
    --dtype float16