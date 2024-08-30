import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, set_start_method

WORLD_SIZE = 8
REPEAT = 100
WARMUP_ITERS = 5
OP = "allreduce"

SHAPES = [
    [8, 1536],
    [8, 4608],
    [8, 6144],
]


def test_op(
    rank,
    result_queue,
    global_shape,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    os.environ["RANK"] = str(rank)

    # Initialize
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=WORLD_SIZE,
        device_id=device,
    )
    torch.cuda.set_device(device)

    start_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(REPEAT)
    ]
    end_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(REPEAT)
    ]

    for idx in range(REPEAT):
        if OP == "allgather":
            output_tensor = torch.zeros(global_shape, device=device)
            local_shape = [global_shape[0], global_shape[1] // WORLD_SIZE]
            local_tensor = torch.randn(local_shape, device=device)

            dist.barrier()
            start_event[idx].record()

            dist.all_gather_into_tensor(output_tensor, local_tensor)
            output_tensor = output_tensor.movedim(0, 1)
            output_tensor = output_tensor.reshape(global_shape)

        elif OP == "allreduce":
            local_tensor = torch.randn(global_shape, device=device)

            dist.barrier()
            start_event[idx].record()

            dist.all_reduce(local_tensor)

        dist.barrier()
        end_event[idx].record()

        torch.cuda.synchronize(device=device)

    elapsed = [start_event[idx].elapsed_time(end_event[idx]) for idx in range(REPEAT)]

    avg = sum(elapsed[WARMUP_ITERS:]) / len(elapsed[WARMUP_ITERS:])

    if rank == 0:
        result_queue.put(avg)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # Set the start method to 'spawn'
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # In case it's already set

    shape = SHAPES[0]
    result_queue = Queue()
    mp.spawn(
        test_op,
        args=(
            result_queue,
            shape,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )

    result = result_queue.get()
    print(f"Shape: {shape}, time: {result}")

    torch.cuda.empty_cache()
