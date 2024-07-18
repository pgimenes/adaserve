export CHECKPOINT=facebook/opt-1.3b
export TENSOR_PARALLEL=8
export BATCH_SIZE=64
export SEQUENCE_LENGTH=128

python src/vllm_bench.py \
    --model_name $CHECKPOINT \
    --tensor_parallel $TENSOR_PARALLEL \
    --batch_size $BATCH_SIZE \
    --input_sequence_length $SEQUENCE_LENGTH