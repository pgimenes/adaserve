import os
import time
import pandas as pd

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate, Shard

from chop.distributed.tensor import distribute_tensor


def device_fn(
    rank,
    world_size,
    device_mesh=None,
):
    """
    This function gets called on each GPU device to set up the distributed environment and distribute the model,
    following the SPMD model.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)

    # Initialize process group
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    # Initialize DeviceMesh
    mesh = DeviceMesh(
        "cuda",
        device_mesh,
    )

    df = pd.read_csv("src/datasets/AzureLLMInferenceTrace_conv_parsed.csv")
    df = df[:50]

    start_time = time.time()
    last_time = 0
    while True:
        elapsed_time = time.time() - start_time

        if elapsed_time > df.iloc[-1]["TIMESTAMP"]:
            break

        # Step 1: Send/receive tensor size
        if rank == 0:

            # Filter rows between last_time and elapsed_time
            filtered_df = df[df["TIMESTAMP"] < elapsed_time]
            filtered_df = filtered_df[filtered_df["TIMESTAMP"] >= last_time]

            # formulate a batch
            bsz = len(filtered_df)
            seq_len = filtered_df["ContextTokens"].max()
            max_out_tokens = filtered_df["GeneratedTokens"].max()

            if bsz == 0:
                continue

            inputs = torch.randn(
                (
                    bsz,
                    seq_len,
                    1536,
                )
            ).to(device)

            tensor_size = torch.tensor(
                inputs.shape,
                dtype=torch.long,
            ).to(device)
        else:
            tensor_size = torch.empty(
                3,
                dtype=torch.long,
            ).to(device)

        # Broadcast the tensor size to all GPUs
        dist.broadcast(tensor_size, src=0)
        dist.barrier()

        # if rank == 1:
        #     print(f"Rank {rank} received tensor size {tensor_size}")

        # Step 2: Create and receive the actual tensor
        if rank != 0:
            inputs = torch.empty(tuple(tensor_size.tolist())).to(device)

        dist.broadcast(inputs, src=0)
        dist.barrier()
        if rank == 1:
            print(f"Rank {rank} received tensor of shape {inputs.shape}")

        # Distribute tensor across devices
        distributed_inputs = distribute_tensor(
            inputs,
            mesh,
            (Replicate(), Shard(0)),
        )

        # Do some computation
        time.sleep(1)

        # Reset last_time (this only runs if we ran some computation)
        last_time = elapsed_time

    # Clean up
    dist.destroy_process_group()


# Define run parameters
world_size = 8
device_mesh = torch.arange(world_size).reshape(2, 4)  # Initialize device mesh

if __name__ == "__main__":
    mp.spawn(
        device_fn,
        args=(
            world_size,
            device_mesh,
        ),
        nprocs=world_size,
        join=True,
    )
