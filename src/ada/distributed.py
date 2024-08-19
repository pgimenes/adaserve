import os
import torch
from functools import partial
import pandas as pd
import time

import torch.distributed as dist

from torch.distributed._tensor import DeviceMesh

from chop.distributed.tensor import distribute_module, distribute_tensor, DTensor
from chop.distributed.utils import (
    rlog,
    distributed_timing,
    distributed_average_timing,
    dist_model_fn,
)
import chop.passes as passes
from chop.passes.graph.analysis.utils import fetch_attr, load_arg
from chop.tools import get_logger

from ada.auto import autosharding_runner

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def node_interpreter(rank, mg, inputs):
    """
    Execute the graph node-by-node following the interpreter pattern
    https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern

    Args:
        rank (int): Rank of the current process.
        mg (MaseGraph): input MaseGraph
        inputs (DTensor): input distributed tensor
    """
    env = {}
    for node in mg.fx_graph.nodes:
        args, kwargs = None, None
        if node.op == "placeholder":
            result = inputs
            rlog(
                logger,
                rank,
                f"Placeholder {node.name} has type {type(inputs)}, shape {inputs.shape}",
                level="debug",
            )

        elif node.op == "get_attr":
            result = fetch_attr(mg.model, node.target)
            rlog(
                logger,
                rank,
                f"get_attr node {node.name} has type {type(inputs)}, shape {inputs.shape}",
                level="debug",
            )
            if isinstance(result, DTensor):
                rlog(
                    logger,
                    rank,
                    f"Local shape: {result._local_tensor.shape}, sharding: {result._spec}",
                    level="debug",
                )

        elif node.op == "call_function":
            args = load_arg(node.args, env)
            kwargs = load_arg(node.kwargs, env)
            rlog(logger, rank, f"Running function {node.name}", level="info")

            for arg_idx, arg in enumerate(args):
                if isinstance(arg, DTensor):
                    rlog(
                        logger,
                        rank,
                        f"Arg {arg_idx} (DTensor) shape: {arg.shape}, local_tensor.shape: {arg._local_tensor.shape}, spec: {arg._spec}",
                        level="debug",
                    )
                elif isinstance(arg, torch.Tensor):
                    rlog(
                        logger,
                        rank,
                        f"Arg {arg_idx} (Tensor) shape: {arg.shape}",
                        level="debug",
                    )
                else:
                    rlog(
                        logger,
                        rank,
                        f"Arg {arg_idx}: {arg}",
                        level="debug",
                    )

            try:
                result = node.target(*args, **kwargs)
            except:
                torch.distributed.breakpoint(0)

            rlist = (result,) if not isinstance(result, (tuple, list)) else result
            for ridx, r in enumerate(rlist):
                if isinstance(r, DTensor):
                    rlog(
                        logger,
                        rank,
                        f"Result {ridx} (DTensor) shape: {r.shape}, local_shape: {r._local_tensor.shape}, spec: {r._spec}",
                        level="debug",
                    )

                    if node.meta["mase"]["common"].get("results", None) is not None:
                        node_spec = node.meta["mase"]["common"]["results"][
                            f"data_out_{ridx}"
                        ]["dtensor_spec"]
                        rlog(
                            logger,
                            rank,
                            f"expected spec: {node_spec}",
                            level="debug",
                        )
                elif isinstance(r, torch.Tensor):
                    rlog(
                        logger,
                        rank,
                        f"Result {ridx} (Tensor) shape: {r.shape}",
                        level="debug",
                    )
                else:
                    rlog(
                        logger,
                        rank,
                        f"Result {ridx}: {r}",
                        level="debug",
                    )

        # Register in environment dictionary
        env[node.name] = result


def single_batch_device_fn(
    rank,
    world_size,
    device_mesh=None,
    model_class=None,
    model_config=None,
    cli_args=None,
):
    """
    This function gets called on each GPU device to set up the distributed environment and distribute the model,
    following the SPMD model.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)

    # Initialize
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    # Run the autosharding pass etc
    inputs = {
        "input_ids": torch.randint(
            0,
            50256,
            (
                cli_args.batch_size,
                cli_args.sequence_length,
            ),
        ),
        "position_ids": torch.arange(
            0,
            cli_args.sequence_length,
            dtype=torch.long,
        ),
        "kv_caches": torch.randn(10),
        "attn_metadata": torch.randn(10),
        "intermediate_tensors": torch.randn(10),
    }
    mg, pass_outputs = autosharding_runner(
        model_class=model_class,
        model_config=model_config,
        cli_args=cli_args,
        inputs=inputs,
    )

    # Distribute model parameters according to sharding configuration
    mesh = DeviceMesh("cuda", mesh=device_mesh)
    rlog(logger, rank, f"Distributing module parameters...", level="info")
    mg.model, dist_time = distributed_timing(
        distribute_module,
        mg.model,
        mesh,
        partial(
            dist_model_fn,
            rank=rank,
            tensor_sharding_map=pass_outputs["autosharding_analysis_pass"][
                "tensor_sharding_map"
            ],
        ),
        input_fn=None,
        output_fn=None,
    )
    rlog(logger, rank, f"Module distribution done. Time taken: {dist_time} seconds.")

    # Run forward pass
    rlog(logger, rank, f"Starting forward pass.", level="info")

    # Get the sharding spec for the input tensor
    for node in mg.fx_graph.nodes:
        if node.name == "input_ids":
            input_ids_sharding = node.meta["mase"]["common"]["results"]["data_out_0"][
                "dtensor_spec"
            ].placements

        if node.name == "position_ids":
            position_ids_sharding = node.meta["mase"]["common"]["results"][
                "data_out_0"
            ]["dtensor_spec"].placements

    rlog(logger, rank, f"Sharding input to: {input_ids_sharding}", level="debug")
    input_ids = distribute_tensor(
        inputs["input_ids"],
        mesh,
        input_ids_sharding,
    )
    position_ids = distribute_tensor(
        inputs["position_ids"],
        mesh,
        position_ids_sharding,
    )

    if cli_args.debug:
        node_interpreter(rank, mg, inputs)
    else:
        _, time_taken = distributed_average_timing(
            fn=mg.model,
            args={
                "input_ids": input_ids,
                "position_ids": position_ids,
                "kv_caches": inputs["kv_caches"],
                "attn_metadata": inputs["attn_metadata"],
                "intermediate_tensors": inputs["intermediate_tensors"],
            },
            repeat=4,
            warmup_iters=2,
        )
        rlog(
            logger,
            rank,
            f"Forward pass finished. Time taken: {time_taken}",
            level="info",
        )

    dist.destroy_process_group()


def serving_device_fn(
    rank,
    world_size,
    device_mesh=None,
    model_class=None,
    model_config=None,
    cli_args=None,
):
    """
    This function gets called on each GPU device to set up the distributed environment and distribute the model,
    following the SPMD model.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)

    # Initialize
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    mesh = DeviceMesh(
        "cuda",
        mesh=device_mesh,
    )

    NUM_REQUESTS = 50

    # Run the autosharding pass etc
    mg, pass_outputs = autosharding_runner(
        model_class=model_class,
        model_config=model_config,
        cli_args=cli_args,
    )

    rlog(logger, rank, f"Distributing module parameters...", level="info")
    mg.model, dist_time = distributed_timing(
        distribute_module,
        mg.model,
        mesh,
        partial(
            dist_model_fn,
            rank=rank,
            tensor_sharding_map=pass_outputs["autosharding_analysis_pass"][
                "tensor_sharding_map"
            ],
        ),
        input_fn=None,
        output_fn=None,
    )
    rlog(logger, rank, f"Module distribution done. Time taken: {dist_time} seconds.")

    # Load dataset
    ds = "AzureLLMInferenceTrace_conv_parsed.csv"
    rlog(logger, rank, f"Loading dataset: {ds}", level="info")
    df = pd.read_csv(f"src/ada/datasets/{ds}")
    df = df[:NUM_REQUESTS]

    # Get the sharding spec for the input tensor
    for node in mg.fx_graph.nodes:
        if node.op == "placeholder":
            input_sharding = node.meta["mase"]["common"]["results"]["data_out_0"][
                "dtensor_spec"
            ].placements

    start_time = time.time()
    left_time_ptr = 0
    jcts = []
    while True:
        right_time_ptr = time.time() - start_time

        if left_time_ptr > df.iloc[NUM_REQUESTS - 1]["TIMESTAMP"]:
            break

        # In rank 0, load batch from dataset using two pointer traversal
        if rank == 0:
            # Filter rows between left_time_ptr and right_time_ptr
            batch_df = df[df["TIMESTAMP"] < right_time_ptr]
            batch_df = batch_df[batch_df["TIMESTAMP"] >= left_time_ptr]

            # Formulate a batch
            bsz = len(batch_df)
            seq_len = batch_df["ContextTokens"].max()

            # Exit if no requests are loaded yet
            if bsz == 0:
                continue

        # Synchronize with all processes to update the left_time_ptr
        # which indicates the time at which computation starts
        dist.barrier()
        left_time_ptr = right_time_ptr

        # Step 1: Send/receive tensor size
        if rank == 0:
            random_batch = torch.randn(
                (
                    bsz,
                    seq_len,
                    model_config.hidden_size,
                )
            ).to(device)

            tensor_size = torch.tensor(
                random_batch.shape,
                dtype=torch.long,
            ).to(device)

            max_tokens_per_sequence = torch.tensor(
                batch_df["GeneratedTokens"].tolist(),
                dtype=torch.long,
            ).to(device)

            logger.debug(f"Formulated batch of size {random_batch.shape}")
        else:
            tensor_size = torch.empty(
                3,
                dtype=torch.long,
            ).to(device)

        # Broadcast the tensor size to all GPUs
        dist.broadcast(tensor_size, src=0)
        dist.barrier()

        # Step 2: Create and receive the actual tensor
        if rank != 0:
            sizes = tensor_size.tolist()
            random_batch = torch.empty(tuple(sizes)).to(device)
            max_tokens_per_sequence = torch.empty(
                sizes[0],
                dtype=torch.long,
            ).to(device)

        dist.broadcast(random_batch, src=0)
        dist.broadcast(max_tokens_per_sequence, src=0)
        dist.barrier()

        rlog(logger, rank, f"Sharding input to: {input_sharding}", level="debug")
        distributed_inputs = distribute_tensor(
            random_batch,
            mesh,
            input_sharding,
        )

        try:
            if cli_args.debug:
                node_interpreter(
                    rank,
                    mg,
                    distributed_inputs,
                )
            else:
                _ = mg.model(distributed_inputs)

                # Generation loop
                # Currently not using KV cache
                generated_tokens = 1
                bsz = tensor_size[0]
                out_tokens = max(max_tokens_per_sequence)

                while generated_tokens < out_tokens:
                    new_tokens = torch.randn(
                        (
                            bsz,
                            1,
                            model_config.hidden_size,
                        ),
                        device=device,
                    )
                    concat_out = torch.cat((random_batch, new_tokens), dim=1)
                    distributed_inputs = distribute_tensor(
                        concat_out,
                        mesh,
                        input_sharding,
                    )
                    out = mg.model(distributed_inputs)

                    generated_tokens += 1
                    rlog(
                        logger,
                        rank,
                        f"Generated {generated_tokens} / {out_tokens} tokens",
                        level="debug",
                    )

                dist.barrier(async_op=True)
                finish_time = time.time() - start_time

                rlog(
                    logger,
                    rank,
                    f"Forward pass finished. Time taken: {finish_time - right_time_ptr}s",
                    level="debug",
                )

                if rank == 0:
                    batch_jcts = (finish_time - batch_df["TIMESTAMP"]).tolist()
                    jcts += batch_jcts

        except torch.OutOfMemoryError:
            rlog(
                logger,
                rank,
                f"Batch went OoM, so all requests will be dropped. This will impact SLA.",
                level="warning",
            )

        except Exception as e:
            rlog(
                logger,
                rank,
                f"Unknown exception while handling batch: {e}.",
                level="error",
            )
            dist.destroy_process_group()

        torch.cuda.empty_cache()

    # Summarize results
    if rank == 0:
        rlog(
            logger,
            rank,
            f"SLA: {len(jcts) / NUM_REQUESTS}, Average JCT: {sum(jcts) / len(jcts)}",
            level="warning",
        )

    dist.destroy_process_group()
