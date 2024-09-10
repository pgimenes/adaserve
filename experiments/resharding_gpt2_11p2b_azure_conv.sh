export CHECKPOINT=$ADASERVE_CHECKPOINTS_PATH/nice-gpt2-11.2b
export TENSOR_PARALLEL=8

ada \
    --dynamic_resharding \
    --model_name $CHECKPOINT \
    --dataset azure_conv \
    --tensor_parallel $TENSOR_PARALLEL