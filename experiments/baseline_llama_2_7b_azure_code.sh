export CHECKPOINT=$ADASERVE_CHECKPOINTS_PATH/meta-llama/Llama-2-7b-hf
export TENSOR_PARALLEL=8

ada \
    --model_name $CHECKPOINT \
    --dataset azure_code \
    --debug \
    --max_requests 50 \
    --tensor_parallel $TENSOR_PARALLEL