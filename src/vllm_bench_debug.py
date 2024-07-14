"""
The simplest benchmark for debugging vllm tp across multi-ndoe
"""

from vllm import LLM, SamplingParams
import argparse
import os
import time
import subprocess

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--huggingface_cache", type=str
    )  # use specific huggingface cache if specified
    parser.add_argument(
        "--model_name", type=str, required=True
    )  # huggingface model name
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensor_parallel", type=int, default=1
    )  # this is where tensor parallelism is specified
    parser.add_argument("--input_sequence_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--nic_name", type=str)
    return parser.parse_args()

def get_rdma_perf_counter(nic_name):
    # get rdma performance counter
    command = [
        "ethtool",
        "-S",
        nic_name,
        "|",
        "grep",
        "rx_vport_rdma_unicast_bytes"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    bytes = int(result.stdout.split("\n")[0].split(":")[1])
    Gbytes = bytes / 1024 / 1024 / 1024
    return Gbytes


if __name__ == "__main__":
    args = cli()

    # setup huggingface cache if specified
    if args.huggingface_cache:
        os.environ["HF_HOME"] = args.huggingface_cache

    # figure out the way to use hf to directly load model and tokenizer(see documnet example)

    # Load model
    if args.tensor_parallel > 1:
        model = LLM(
            model=args.model_name,
            tensor_parallel_size=args.tensor_parallel,
            seed=args.seed,
        )
    else:
        model = LLM(model=args.model_name, seed=args.seed)

    # generate the fake dataset
    # load test prompt from prompt.txt for now
    with open("prompt.txt", "r") as f:
        prompt = f.read()
    prompt = prompt[
        : args.input_sequence_length
    ]  # truncate based on tokenizer instead?
    prompts = [prompt] * args.batch_size

    sampling_params = SamplingParams(
        seed=args.seed,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1,  # for some reason it seems to act strangely when token < 4, need to double check how to get max_tokens = 1 to work
    )

    elapsed_times = []
    used_bw_gbs = []

    for repeat in range(args.repeat):
        start_time = time.time()
        start_gb = get_rdma_perf_counter(args.nic_name)
        generated_outputs = model.generate(prompts, sampling_params=sampling_params)
        end_time = time.time()
        end_gb = get_rdma_perf_counter(args.nic_name)
        elapsed_times.append(end_time - start_time)
        used_bw_gbs.append((end_gb - start_gb)/(end_time - start_time))

    elapsed_time = sum(elapsed_times) / len(elapsed_times)
    average_bw_gbs = sum(used_bw_gbs) / len(used_bw_gbs)
    
    print(
        "model: {}, GPU Num: {}, input_sequence_length: {}, batch_size: {}, elapsed_time: {}".format(
            args.model_name,
            args.tensor_parallel,
            args.input_sequence_length,
            args.batch_size,
            elapsed_time,
        )
    )
    print("average rdma bandwidth: {} GB/s".format(average_bw_gbs))