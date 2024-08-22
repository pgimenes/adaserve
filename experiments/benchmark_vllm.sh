export CHECKPOINT=$ADASERVE_CHECKPOINTS_PATH/nice-gpt2-1.5b
export TENSOR_PARALLEL=$(nvidia-smi --list-gpus | wc -l)
export BATCH_SIZE=8
export SEQUENCE_LENGTH=128

python src/ada/vllm_bench.py \
    --model_name $CHECKPOINT \
    --tensor_parallel $TENSOR_PARALLEL \
    --batch_size $BATCH_SIZE \
    --input_sequence_length $SEQUENCE_LENGTH