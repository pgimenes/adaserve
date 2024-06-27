import os
import time
from functools import partial

import torch
import torch.distributed as dist
from chop.distributed.debug import visualize_sharding
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)

WORLD_SIZE = 8


# from models.bert.configuration_bert import BertConfig
# from models.bert.modeling_bert import BertModel

class MLP(nn.Module):
    def __init__(self, inf, outf):
        super().__init__()
        self.layer = nn.Linear(inf, outf)

    def forward(self, x):
        x = x.expand(5, -1, -1)
        return self.layer(x)

def pprint(rank, msg):
    if (rank == 0):
        print(msg)

def deepsetattr(obj, attr, value):
    """Recurses through an attribute chain to set the ultimate value."""
    attrs = attr.split(".")
    if len(attrs) > 1:
        deepsetattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]), value)
    else:
        setattr(obj, attr, value)

def pvisualize_sharding(rank, module):
    if (rank == 0):
        visualize_sharding(module)

def dist_model_fn(
    name: str, module: nn.Module, device_mesh: DeviceMesh, rank: int, weight_sharding=None
) -> None:
    pprint(rank, f"Processing module {module}")

    for name, param in module.named_parameters():
        pprint(rank, f"Has parameter {name} of shape {param.shape}")
        if ("weight" in name):
            pprint(rank, f"Processing parameter {name} of type {type(param)}")
            param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, weight_sharding)
            )
            deepsetattr(module, name, param)


def test_dist(rank, world_size, input_sharding = None, weight_sharding=None):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(1.0, device)
    mesh = DeviceMesh("cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])

    torch.manual_seed(12)

    # config = BertConfig()
    # config.num_hidden_layers = 1
    config_sequence_length = 128
    # model = BertModel(config).to(device)
    model = MLP(64, 64).to(device)

    model = distribute_module(
        model, mesh, partial(dist_model_fn, rank=rank, weight_sharding=weight_sharding), input_fn=None, output_fn=None
    )

    # in_data = torch.randn((1, config_sequence_length, config.hidden_size)).to(device)
    in_data = torch.randn((64, 64)).to(device)
    in_data = distribute_tensor(in_data, mesh, input_sharding)

    tensor_map = {"in_data": in_data}

    pprint(rank, f"\n================= RUNNING.... \n")
    out = model(in_data)
    tensor_map["out"] = out

    for name, tensor in tensor_map.items():
        pprint(rank, f"============== {name} ==============")
        try:
            pvisualize_sharding(rank, tensor)
        except Exception as e:
            pprint(rank, f"Failed to visualize {name} with {e}")
        pprint(rank, "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    import itertools
    opts = itertools.product([Replicate(), Shard(0), Shard(1)], repeat=2)
    # opts = [(Shard(0), Replicate())]

    success, fail = [], []
    # for sharding in itertools.product(opts, repeat=2):
    for sharding in opts:
        print(f"Running: {sharding}")
        try:
            mp.spawn(partial(test_dist, input_sharding=(Replicate(), Replicate()), weight_sharding=sharding), args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
            success.append(sharding)
        except Exception as e:
            print(f"Failed: {sharding} with {e}")
            fail.append(sharding)
            continue

    print(f"========= SUMMARY ========")
    print(f"Success: {success}")
    print(f"Failed: {fail}")