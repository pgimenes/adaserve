export CHECKPOINT=$ADASERVE_CHECKPOINTS_PATH/nice-gpt2-4b
export TENSOR_PARALLEL=$(nvidia-smi --list-gpus | wc -l)
export BATCH_SIZE=32
export SEQUENCE_LENGTH=128

export VLLM_AUTOSHARDING_DATA_SIZE=1152

python src/ada/single_batch.py \
    --model_name $CHECKPOINT \
    --tensor_parallel $TENSOR_PARALLEL \
    --batch_size $BATCH_SIZE \
    --input_sequence_length $SEQUENCE_LENGTH