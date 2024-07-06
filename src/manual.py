import os
import itertools
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed._tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)


def pprint(rank, msg):
    if rank == 0:
        print(msg)


def deepsetattr(obj, attr, value):
    attrs = attr.split(".")
    if len(attrs) > 1:
        deepsetattr(getattr(obj, attrs[0]), ".".join(attrs[1:]), value)
    else:
        setattr(obj, attr, value)


def dist_model_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
    rank: int,
    weight_sharding=None,
) -> None:
    pprint(rank, f"Processing module {module}")
    for name, param in module.named_parameters():
        if name in ["wte.weight", "wpe.weight"]:
            return
        if "weight" in name:
            pprint(
                rank,
                f"Processing parameter {name} of type {type(param)} with shape {param.shape}",
            )
            if len(param.shape) < 2:
                pprint(rank, f"Skipping 1d parameter {name}")
                return
            param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, weight_sharding)
            )
            deepsetattr(module, name, param)


def device_fn(
    rank,
    world_size,
    config,
    model_class,
    input_sharding=None,
    weight_sharding=None,
    cli_args=None,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(1.0, device)
    mesh = DeviceMesh("cuda", mesh=cli_args.device_mesh)
    torch.manual_seed(12)
    model = model_class(config).to(device)
    model = distribute_module(
        model,
        mesh,
        partial(dist_model_fn, rank=rank, weight_sharding=weight_sharding),
        input_fn=None,
        output_fn=None,
    )
    in_data = torch.randn(
        (cli_args.batch_size, cli_args.sequence_length, config.hidden_size)
    ).to(device)
    in_data = distribute_tensor(in_data, mesh, input_sharding)
    pprint(rank, f"\n================= RUNNING.... \n")
    _ = model(in_data)
    dist.destroy_process_group()


def manual_sharding_runner(model_class, model_config, args):
    if args.row:
        opts = [(Shard(0), Replicate())]
    elif args.column:
        opts = [(Shard(1), Replicate())]
    else:
        opts = itertools.product([Replicate(), Shard(0), Shard(1)], repeat=2)

    success, fail = [], []
    for sharding in opts:
        print(f"Running: {sharding}")
        try:
            mp.spawn(
                partial(
                    device_fn,
                    config=model_config,
                    model_class=model_class,
                    input_sharding=(Replicate(), Replicate()),
                    weight_sharding=sharding,
                    cli_args=args,
                ),
                args=(args.world_size,),
                nprocs=args.world_size,
                join=True,
            )
            success.append(sharding)
        except Exception as e:
            print(f"Failed: {sharding} with {e}")
            fail.append(sharding)
            continue

    print(f"========= SUMMARY ========")
    print(f"Success: {success}")
    print(f"Failed: {fail}")
