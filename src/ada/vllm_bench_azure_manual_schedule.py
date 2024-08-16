import csv
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import argparse
import time
from dataclasses import dataclass

import logging

logger = logging.getLogger("vllm_bench")
logging.basicConfig(level=logging.INFO)

DATASET_PATH = "src/ada/datasets/AzureLLMInferenceTrace_conv_parsed.csv"

@dataclass
class Request:
    id: int
    receive_timestamp: float
    context_tokens: int
    response_tokens: int
    finish_timestamp: float = 0.0
    

    @classmethod
    def from_csv_row(cls, row):
        return Request(
            id=int(row[0]),
            receive_timestamp=float(row[1]),
            context_tokens=int(row[2]),
            response_tokens=int(row[3]),
        )

def load_dataset(path):
    # load 
    dataset = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        for row in csv_reader:
            dataset.append(Request.from_csv_row(row))
    return dataset

def dump_results(dataset, path):
    with open(path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id", "receive_timestamp", "context_tokens", "response_tokens", "finish_timestamp"])
        for request in dataset:
            csv_writer.writerow([
                request.id,
                request.receive_timestamp,
                request.context_tokens,
                request.response_tokens,
                request.finish_timestamp,
            ])

def make_request_batches(dataset : list, args):
    # devide dataset criteria
    # 1. number requests
    # 2. time window 
    # 3. max input tokens

    result = []
    
    batch = []
    n_reqs = 0
    prev_time = 0.0
    cur_tokens = 0
    make_batch_flag = False

    while len(dataset) > 0:
        request = dataset.pop(0)
        n_reqs += 1
        cur_tokens += request.context_tokens
        diff_time = request.receive_timestamp - prev_time
        batch.append(request)

        if n_reqs >= args.max_requests:
            make_batch_flag = True
        if diff_time >= args.time_window:
            make_batch_flag = True
        if cur_tokens >= args.max_input_tokens:
            make_batch_flag = True

        if make_batch_flag:
            result.append(batch)
            batch = []
            n_reqs = 0
            prev_time = request.receive_timestamp
            cur_tokens = 0
            make_batch_flag = False
    return result


def make_token_batch(request_batch, tokenizer):
    with open("prompt.txt", "r") as f:
        prompt = f.read()
    token_prompt = TokensPrompt(prompt_token_ids= tokenizer.encode(prompt))
    token_batch = []
    max_response_tokens = 0
    for request in request_batch:
        token_batch.append({
            "prompt_token_ids": token_prompt["prompt_token_ids"][:request.context_tokens]
        })
        max_response_tokens = max(max_response_tokens, request.response_tokens)
    return token_batch, max_response_tokens

def measure_batch_latency(model, token_batch, max_response_tokens):
    sample_params = SamplingParams(
        max_tokens=max_response_tokens
    )

    start_time = time.time()
    result = model.generate(
        prompts=token_batch,
        sampling_params=sample_params,
    )
    logger.debug("Result", result)
    print(result)
    end_time = time.time()
    
    return end_time - start_time


def measure_total_time(model, dataset, tokenizer, args):
    request_batches = make_request_batches(dataset, args)

    curr_time = 0.0
    for request_batch in request_batches:
        logger.info(f"Batch {request_batch}")
        token_batch, max_response_tokens = make_token_batch(request_batch, tokenizer)
        batch_issue_time = max(request_batch[-1].receive_timestamp, curr_time)
        logger.info(f"Issue time {batch_issue_time}")
        logger.info(f"max_respose_tokens {max_response_tokens}")
        latency = measure_batch_latency(model, token_batch, max_response_tokens)
        logger.info(f"Latency {latency}")
        curr_time = batch_issue_time + latency
        for request in request_batch:
            request.finish_timestamp = curr_time
    return dataset

def load_model(args) -> LLM:
        # Load model
    dtype = torch.float32
    if args.datatype == "float16":
        dtype = torch.float16
    if args.datatype == "bfloat16":
        dtype = torch.bfloat16
    if args.tensor_parallel > 1:
        model = LLM(
            model=args.model_name,
            tensor_parallel_size=args.tensor_parallel,
            seed=args.seed,
            enforce_eager=True,
            trust_remote_code=True,
            dtype=dtype,
        )
    else:
        model = LLM(
            model=args.model_name,
            seed=args.seed,
            enforce_eager=True,
            dtype=dtype,
        )
    return model


def evaluate(args):
    logger.info(f"Evaluating model: {args.model_name}")
    logger.info(f"Tensor Parallel: {args.tensor_parallel}")


    model = load_model(args)
    dataset = load_dataset(args.dataset_path)
    tokenizer = model.llm_engine.get_tokenizer()

    dataset = measure_total_time(model, dataset, tokenizer, args)

    for i in dataset:
        print(i)


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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="float32",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DATASET_PATH,
    )
    parser.add_argument(
        "--max_requests",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--time_window",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.csv",
    )



    return parser.parse_args()

if __name__ == "__main__":
    args = cli()
    main(args)
