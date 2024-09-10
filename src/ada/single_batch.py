import torch
from vllm import LLM, SamplingParams
import argparse
import os
import time
import subprocess

# define logger
from viztracer import VizTracer
import logging

logger = logging.getLogger("vllm_bench")
logging.basicConfig(level=logging.INFO)


def get_rdma_perf_counter(nic_name):
    # get rdma performance counter
    command = [
        "ethtool",
        "-S",
        nic_name
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    # find the line with rx_vport_rdma_unicast_bytes
    output = result.stdout.split("\n")
    output = [line for line in output if "rx_vport_rdma_unicast_bytes" in line][0]
    bytes = int(output.split(":")[1].strip())
    Gbytes = bytes / 1024 / 1024 / 1024
    return Gbytes


def evaluate(args):
    logger.info(f"Evaluating model: {args.model_name}")
    logger.info(f"Tensor Parallel: {args.tensor_parallel}")
    logger.info(f"Input Sequence Length: {args.input_sequence_length}")
    logger.info(f"Batch Size: {args.batch_size}\n")

    # setup huggingface cache if specified
    if args.huggingface_cache:
        os.environ["HF_HOME"] = args.huggingface_cache

    # sharding config
    prefill_sharding = {}
    for layer in range(96):
        prefill_sharding[f"transformer.h.{layer}.ln_1"] = "replicated"
        prefill_sharding[f"transformer.h.{layer}.attn.c_attn"] = "column"
        prefill_sharding[f"transformer.h.{layer}.attn.attn"] = "head"
        prefill_sharding[f"transformer.h.{layer}.attn.c_proj"] = "row"
        prefill_sharding[f"transformer.h.{layer}.res_1"] = "replicated"
        prefill_sharding[f"transformer.h.{layer}.ln_2"] = "data"
        prefill_sharding[f"transformer.h.{layer}.mlp.c_fc"] = "data"
        prefill_sharding[f"transformer.h.{layer}.mlp.c_proj"] = "data"
        prefill_sharding[f"transformer.h.{layer}.res_2"] = "data"
        prefill_sharding[f"transformer.ln_f"] = "replicated"

    decode_sharding = {}
    for layer in range(96):
        decode_sharding[f"transformer.h.{layer}.ln_1"] = "replicated"
        decode_sharding[f"transformer.h.{layer}.attn.c_attn"] = "column"
        decode_sharding[f"transformer.h.{layer}.attn.attn"] = "head"
        decode_sharding[f"transformer.h.{layer}.attn.c_proj"] = "row"
        decode_sharding[f"transformer.h.{layer}.res_1"] = "replicated"
        decode_sharding[f"transformer.h.{layer}.ln_2"] = "replicated"
        decode_sharding[f"transformer.h.{layer}.mlp.c_fc"] = "column"
        decode_sharding[f"transformer.h.{layer}.mlp.c_proj"] = "row"
        decode_sharding[f"transformer.h.{layer}.res_2"] = "replicated"
        decode_sharding[f"transformer.ln_f"] = "replicated"

    # Load model
    if args.tensor_parallel > 1:
        model = LLM(
            model=args.model_name,
            tensor_parallel_size=args.tensor_parallel,
            seed=args.seed,
            enforce_eager=True,
            trust_remote_code=True,
            dtype=torch.float32,
            load_format="mase",
            prefill_sharding=prefill_sharding,
            decode_sharding=decode_sharding,
        )
    else:
        model = LLM(
            model=args.model_name,
            seed=args.seed,
            enforce_eager=True,
            dtype=torch.float32,
        )

    # generate the fake dataset
    # load test prompt from prompt.txt for now
    with open("experiments/prompt.txt", "r") as f:
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
        detokenize=False,
    )

    elapsed_times = []
    used_bw_gbs = []

    for itr in range(args.repeat):
        logger.info(f"Running iteration: {itr}")
        start_gb = get_rdma_perf_counter(args.nic_name)
        start_time = time.time()
        _ = model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        end_time = time.time()
        end_gb = get_rdma_perf_counter(args.nic_name)
        elapsed_times.append(end_time - start_time)
        used_bw_gbs.append((end_gb - start_gb)/(end_time - start_time))
        
        logger.info(f"Time taken: {end_time - start_time}s")

    elapsed_time = sum(elapsed_times[2:]) / len(elapsed_times[2:])
    average_bw_gbs = sum(used_bw_gbs) / len(used_bw_gbs)
    tps = (args.batch_size * args.input_sequence_length) / elapsed_time

    print(
        "[EVALUATION FINISHED]: model: {}, TP: {}, input_sequence_length: {}, batch_size: {}, elapsed_time: {}, tps: {}".format(
            args.model_name,
            args.tensor_parallel,
            args.input_sequence_length,
            args.batch_size,
            elapsed_time,
            tps,
        )
    )
    print("average rdma bandwidth: {} GB/s".format(average_bw_gbs))

    return elapsed_time, tps


def main(args):
    evaluate(args)


def cli():
    parser = argparse.ArgumentParser()

    # Main parameters
    parser.add_argument(
        "--model_name",
        type=str,
    )
    parser.add_argument(
        "--tensor_parallel",
        type=int,
        default=1,
    )

    # Environment
    parser.add_argument("--huggingface_cache", type=str, default="/data/huggingface/")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--nic_name",
        type=str,
        default="enp195s0f0np0"
        )

    # Evaluation parameters
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--input_sequence_length",
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
    )

    return parser.parse_args()


tracer = VizTracer()
tracer.start()

if __name__ == "__main__":
    args = cli()
    main(args)

tracer.stop()
tracer.save()
