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
    actual_response_tokens: int = 0

    @classmethod
    def from_csv_row(cls, row):
        return Request(
            id=int(row[0]),
            receive_timestamp=float(row[1]),
            context_tokens=int(row[2]),
            response_tokens=int(row[3]),
        )


def load_dataset(path):
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
        csv_writer.writerow(
            [
                "id",
                "receive_timestamp",
                "context_tokens",
                "response_tokens",
                "finish_timestamp",
                "actual_response_tokens",
            ]
        )
        for request in dataset:
            csv_writer.writerow(
                [
                    request.id,
                    request.receive_timestamp,
                    request.context_tokens,
                    request.response_tokens,
                    request.finish_timestamp,
                    request.actual_response_tokens,
                ]
            )


with open("experiments/prompt.txt", "r") as f:
    prompt = f.read()


def to_tokens(length, tokenizer):
    token_prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(prompt))
    return {"prompt_token_ids": token_prompt["prompt_token_ids"][:length]}


def process_all(model, dataset, args):
    engine = model.llm_engine
    tokenizer = engine.get_tokenizer()
    id = 0

    engine_requests = []

    for request in dataset:
        request = dataset[id]
        tokens = to_tokens(request.context_tokens, tokenizer)
        sampling_params = SamplingParams(
            max_tokens=request.response_tokens,
        )
        print(f"Adding request {id}")
        engine_requests.append(
            (
                id,
                tokens,
                sampling_params,
                request.receive_timestamp,
            )
        )
        id += 1

    base_time = time.time()
    print(f"Base time {base_time}")

    while True:
        if engine_requests:
            id, tokens, sampling_params, td = engine_requests[0]
            cur_time = time.time()
            cur_td = cur_time - base_time
            if td <= cur_td:
                print(f"Adding request {id} at {cur_td}, {td}")
                engine.add_request(str(id), tokens, sampling_params)
                engine_requests.pop(0)

        request_output = engine.step()
        for o in request_output:
            if o.finished:
                dataset[int(o.request_id)].actual_response_tokens = len(
                    o.outputs[0].token_ids
                )
                dataset[int(o.request_id)].finish_timestamp = (
                    o.metrics.finished_time - base_time
                )
                dataset[int(o.request_id)].jct = dataset[int(o.request_id)].finish_timestamp - dataset[int(o.request_id)].receive_timestamp

        if not (engine.has_unfinished_requests() or engine_requests):
            break

    return dataset

# sharding config
prefill_sharding = {}
for layer in range(24):
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
            load_format="mase", # enable Mase model optimizations when loading the model
            enable_dynamic_resharding=True, # enable dynamic resharding
            sharding_config=prefill_sharding,
            prefill_sharding=prefill_sharding,
            decode_sharding=decode_sharding,
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
    dataset = process_all(model, dataset, args)

    dump_results(dataset, args.output_path)

    jct = 0
    for request in dataset:
        jct += request.jct
    jct /= len(dataset)
    logger.info(f"Average JCT: {jct}")


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
    # parser.add_argument("--huggingface_cache", type=str, default="/data/huggingface/")
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
