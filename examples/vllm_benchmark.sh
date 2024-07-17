python src/vllm_bench.py \
    --model_name facebook/opt-1.3b \
    --tensor_parallel 8 \
    --batch_size 136 \
    --input_sequence_length 128