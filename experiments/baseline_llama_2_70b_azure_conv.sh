export CHECKPOINT=$ADASERVE_CHECKPOINTS_PATH/meta-llama/Llama-2-70b-chat-hf
export TENSOR_PARALLEL=8

ada \
    --model_name $CHECKPOINT \
    --dataset azure_conv \
    --tensor_parallel $TENSOR_PARALLEL