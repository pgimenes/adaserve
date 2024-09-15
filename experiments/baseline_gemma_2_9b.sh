export CHECKPOINT=/home/pedrogimenes/huggingface/google/gemma-2-9b-it
export TENSOR_PARALLEL=4

vllm serve \
    $CHECKPOINT \
    --api_key test \
    --load_format mase \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --enforce_eager \
    --dtype float16 \
    --max_num_batched_tokens 16384 \
    --max_model_len 16384