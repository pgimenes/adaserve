import os
import csv
import time
from dataclasses import dataclass
import logging

import torch

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from ada.cli import get_cli_args
from ada.sharding import get_sharding_configs

logger = logging.getLogger("adaserve")

DATASET_DICT = {
    "azure_conv": "datasets/AzureLLMInferenceTrace_conv_parsed.csv",
    "azure_code": "datasets/AzureLLMInferenceTrace_code_parsed.csv",
}

@dataclass
class Request:
    id: int
    receive_timestamp: float
    context_tokens: int
    response_tokens: int
    finish_timestamp: float = 0.0
    actual_response_tokens: int = 0
    jct: int = None
    ttft: int = None

    @classmethod
    def from_csv_row(cls, row):
        return Request(
            id=int(row[0]),
            receive_timestamp=float(row[1]),
            context_tokens=int(row[2]),
            response_tokens=int(row[3]),
        )


def load_dataset(args):
    
    path = DATASET_DICT[args.dataset]

    dataset = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        for row in csv_reader:
            dataset.append(Request.from_csv_row(row))
    return dataset


def dump_results(dataset, out_name, args):
    ds = dataset[:args.max_requests]
    
    
    with open(f"{out_name}.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
                "id",
                "receive_timestamp",
                "context_tokens",
                "response_tokens",
                "finish_timestamp",
                "actual_response_tokens",
                "jct",
                "ttft",
            ]
        )
        for request in ds:
            csv_writer.writerow(
                [
                    request.id,
                    request.receive_timestamp,
                    request.context_tokens,
                    request.response_tokens,
                    request.finish_timestamp,
                    request.actual_response_tokens,
                    request.jct,
                    request.ttft,
                ]
            )

    jct = 0
    ttft = 0
    for request in ds:
        jct += request.jct
        ttft += request.ttft
    jct /= len(ds)
    ttft /= len(ds)
    logger.info(f"Average JCT: {jct}")
    logger.info(f"Average TTFT: {ttft}")


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

    for request in dataset[:args.max_requests]:
        request = dataset[id]
        tokens = to_tokens(request.context_tokens, tokenizer)
        sampling_params = SamplingParams(
            max_tokens=request.response_tokens,
        )
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

    while True:
        if engine_requests:
            id, tokens, sampling_params, td = engine_requests[0]
            cur_time = time.time()
            cur_td = cur_time - base_time
            if td <= cur_td:
                logger.info(f"Adding request {id} at {cur_td}, {td}")
                engine.add_request(str(id), tokens, sampling_params)
                engine_requests.pop(0)

        request_output = engine.step()
        for o in request_output:
            if o.finished:
                dataset[int(o.request_id)].actual_response_tokens = len(
                    o.outputs[0].token_ids
                )
                dataset[int(o.request_id)].finish_timestamp = (
                    o.metrics.finished_time - o.metrics.arrival_time
                )
                dataset[int(o.request_id)].jct = o.metrics.finished_time - o.metrics.arrival_time
                try:
                    dataset[int(o.request_id)].ttft = o.metrics.first_token_time - o.metrics.arrival_time
                except:
                    dataset[int(o.request_id)].ttft = None
                
        if not (engine.has_unfinished_requests() or engine_requests):
            break

    return dataset


def load_model(args) -> LLM:

    prefill_sharding, decode_sharding = get_sharding_configs(args)

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
            load_format="mase",
            enable_dynamic_resharding=args.dynamic_resharding,
            prefill_sharding=prefill_sharding,
            decode_sharding=decode_sharding,
            # disable_log_stats=False,
        )
    else:
        model = LLM(
            model=args.model_name,
            seed=args.seed,
            enforce_eager=True,
            dtype=dtype,
        )
    return model

def _clean_str(str):
    return str.replace("/", "_").replace(".", "_")

def _setup_env(args):
    if args.tensor_parallel > 8:
        os.environ["GLOO_SOCKET_IFNAME"] = "enp195s0np0"
        os.environ["NCCL_SOCKET_IFNAME"] = "enp195s0np0"
        os.environ["VLLM_HOST_IP"] = "10.250.30.42"
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
        os.environ["VLLM_TRACE_FUNCTION"] = "1"
        os.environ["NCCL_DEBUG"] = "INFO"
        # ulimit -n 4096

def main(args):
    _setup_env(args)

    if os.environ.get("ADASERVE_DEBUG", None) is not None:
        args.max_requests = 50
        args.debug = True

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Evaluating model: {args.model_name}")
    logger.info(f"Tensor Parallel: {args.tensor_parallel}")

    # Set output path
    if args.output_path is None:
        path = "out/resharding" if args.dynamic_resharding else "out/baseline"
        path += f"_{_clean_str(args.model_name)}_{_clean_str(args.dataset)}_tp_{args.tensor_parallel}"
        args.output_path = path
    logger.info(f"Output path: {args.output_path}")

    # Set max reqs
    args.max_requests = 1000000 if args.max_requests is None else args.max_requests

    model = load_model(args)
    dataset = load_dataset(args)
    dataset = process_all(model, dataset, args)

    dump_results(dataset, args.output_path, args)

def entry_point():
    args = get_cli_args()
    main(args)

if __name__ == "__main__":
    entry_point()