import torch
from vllm import LLM, SamplingParams
import argparse
import os
import time
import subprocess

# define logger
import logging

logger = logging.getLogger("vllm_bench")
logging.basicConfig(level=logging.INFO)

from viztracer import VizTracer


def evaluate(args):
    logger.info(f"Evaluating model: {args.model_name}")
    logger.info(f"Tensor Parallel: {args.tensor_parallel}")
    logger.info(f"Input Sequence Length: {args.input_sequence_length}")
    logger.info(f"Batch Size: {args.batch_size}\n")

    # setup huggingface cache if specified
    if args.huggingface_cache:
        os.environ["HF_HOME"] = args.huggingface_cache

    # sharding config
    sharding_config = {}
    for layer in range(48):
        sharding_config[f"transformer.h.{layer}.attn.c_attn"] = "replicated"
        sharding_config[f"transformer.h.{layer}.attn.attn"] = "replicated"
        sharding_config[f"transformer.h.{layer}.attn.c_proj"] = "replicated"
        sharding_config[f"transformer.h.{layer}.mlp.c_fc"] = "replicated"
        sharding_config[f"transformer.h.{layer}.mlp.c_proj"] = "replicated"

    # Load model
    if args.tensor_parallel > 1:
        model = LLM(
            model=args.model_name,
            tensor_parallel_size=args.tensor_parallel,
            seed=args.seed,
            enforce_eager=True,
            trust_remote_code=True,
            dtype=torch.float32,
            sharding_config=sharding_config,
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
        max_tokens=10,  # for some reason it seems to act strangely when token < 4, need to double check how to get max_tokens = 1 to work
        detokenize=False,
    )

    elapsed_times = []

    for itr in range(args.repeat):
        logger.info(f"Running iteration: {itr}")
        start_time = time.time()
        _ = model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        end_time = time.time()
        elapsed_times.append(end_time - start_time)
        logger.info(f"Time taken: {end_time - start_time}s")

    elapsed_time = sum(elapsed_times[2:]) / len(elapsed_times[2:])
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

    return elapsed_time, tps


def sweep_runner(args):
    model_sizes = args.sweep_model_size_range
    input_sequence_lengths = args.sweep_sequence_length_range
    batch_sizes = args.sweep_batch_size_range
    tp_sizes = args.sweep_tp_range

    grid = [
        (model_size, input_sequence_length, batch_size, tp_size)
        for model_size in model_sizes
        for input_sequence_length in input_sequence_lengths
        for batch_size in batch_sizes
        for tp_size in tp_sizes
    ]

    for idx, grid_point in enumerate(grid):
        model_size, input_sequence_length, batch_size, tp_size = grid_point

        logger.info(f"{idx}/{len(grid)}: Running grid point: {grid_point}")

        # Launch the evaluation as a subprocess
        command = [
            "python",
            "src/vllm_bench.py",
            "--model_name",
            f"{args.sweep_model}-{model_size}",
            "--input_sequence_length",
            str(input_sequence_length),
            "--batch_size",
            str(batch_size),
            "--tensor_parallel",
            str(tp_size),
        ]

        log_filename = f"experiments/benchmarking/sweep_{args.sweep_model.replace('/', '-')}-{model_size.replace('.', '_')}_bs_{batch_size}_seq_len_{input_sequence_length}_tp_{tp_size}.log"

        try:
            with open(log_filename, "w") as file:
                _ = subprocess.run(
                    command,
                    check=True,
                    stdout=file,
                    stderr=subprocess.STDOUT,
                )

            # Extract solution
            with open(log_filename, "r") as file:
                lines = file.readlines()
                for line in lines:
                    if "[EVALUATION FINISHED]:" in line:
                        elapsed_time = float(line.split()[-3].replace(",", ""))
                        tps = float(line.split()[-1].replace(",", ""))

            # Write to csv
            with open("experiments/benchmarking/sweep_results.csv", "a") as file:
                file.write(
                    f"{args.sweep_model}-{model_size},{input_sequence_length},{batch_size},{tp_size},{elapsed_time},{tps}\n"
                )

        except Exception as e:
            logger.warning(f"Error running command: {command}")
            logger.warning(f"Error: {e}")


def main(args):

    if args.sweep:
        sweep_runner(args)

    else:
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

    # Sweep arguments
    parser.add_argument(
        "--sweep",
        action="store_true",
    )
    parser.add_argument(
        "-sweep_model",
        type=str,
        default="facebook/opt",
    )
    parser.add_argument(
        "-sweep_model_size_range",
        type=list,
        default=["1.3b", "2.7b", "6.7b", "13b", "30b", "66b"],
    )
    parser.add_argument(
        "--sweep_sequence_length_range",
        type=list,
        default=[128, 256, 512, 1024],
    )
    parser.add_argument(
        "--sweep_batch_size_range",
        type=list,
        default=range(100, 1100, 100),
    )
    parser.add_argument(
        "--sweep_tp_range",
        type=list,
        default=[1, 2, 4, 8],
    )

    return parser.parse_args()


tracer = VizTracer()
tracer.start()

if __name__ == "__main__":
    args = cli()
    main(args)

tracer.stop()
tracer.save()
