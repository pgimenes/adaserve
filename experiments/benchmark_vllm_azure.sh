export CHECKPOINT=/data/huggingface/nice-gpt2-1.5b
export TENSOR_PARALLEL=8

python src/ada/vllm_bench_azure.py \
    --model_name $CHECKPOINT \
    --tensor_parallel $TENSOR_PARALLEL