export CHECKPOINT=$ADASERVE_CHECKPOINTS_PATH/nice-gpt2-5.7b
export TENSOR_PARALLEL=$(nvidia-smi --list-gpus | wc -l)

python src/ada/vllm_bench_azure.py \
    --model_name $CHECKPOINT \
    --tensor_parallel $TENSOR_PARALLEL