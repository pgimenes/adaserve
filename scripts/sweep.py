import os
from copy import copy
import itertools
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path

import numpy as np

from chop.tools import get_logger

from ada.cli import parse_args

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def evaluate_grid_point(args):
    bs, seq_len = args.grid_point
    log_filename = f"experiments/sweep/sweep_{args.model}_layers_{args.num_hidden_layers}_bs_{bs}_seq_len_{seq_len}.log"

    command = [
        "ada",
        "--skip_forward",
        # Model config
        "--model",
        str(args.model),
        "--checkpoint",
        str(args.checkpoint),
        "--batch_size",
        str(bs),
        "--sequence_length",
        str(seq_len),
        # Optimizer args
        "--optimizer_time_limit",
        str(args.optimizer_time_limit),
        "--optimizer_mip_rel_gap",
        str(args.optimizer_mip_rel_gap),
        "--benchmarking_device",
        str(args.thread_offset),
    ]

    logger.info(f"Launching command: {' '.join(command)}")

    try:
        with open(log_filename, "w") as file:
            _ = subprocess.run(
                command,
                check=True,
                stdout=file,
                stderr=subprocess.STDOUT,
            )

    except subprocess.CalledProcessError as e:
        logger.warning(f"Grid point {bs} failed with error: {e}")
        return None


def sweep_runner(args=None):
    # Create experiments output directory if it doesn't exist
    Path("experiments/sweep").mkdir(parents=True, exist_ok=True)

    bs_range = np.linspace(
        start=args.sweep_min_bs,
        stop=args.sweep_max_bs,
        num=args.sweep_grid_size,
    ).astype(int)

    seq_len_range = np.linspace(
        start=args.sweep_min_seq_len,
        stop=args.sweep_max_seq_len,
        num=args.sweep_grid_size,
    ).astype(int)

    grid = [i for i in itertools.product(bs_range, seq_len_range)]

    with ThreadPoolExecutor(max_workers=args.sweep_max_threads) as executor:
        # Keep track of active thread offsets
        active_offsets = set()
        futures = []

        # Launch trials
        for idx, grid_point in enumerate(grid):
            thread_offset = idx % args.sweep_max_threads

            logger.info(
                f"Launching grid point {grid_point} with thread offset {thread_offset}"
            )

            # Wait until the thread_offset is available
            while thread_offset in active_offsets:
                # Check if any futures have completed
                done, _ = concurrent.futures.wait(
                    futures,
                    timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    futures.remove(future)
                    active_offsets.remove(future.thread_offset)

            # Launch new thread
            nargs = copy(args)
            setattr(nargs, "grid_point", grid_point)
            setattr(nargs, "thread_offset", thread_offset)
            future = executor.submit(evaluate_grid_point, nargs)
            future.thread_offset = thread_offset  # Attach thread_offset to the future
            futures.append(future)
            active_offsets.add(thread_offset)

        # Process remaining trial outputs
        results = []
        for future in as_completed(futures):
            try:
                res = future.result()
                logger.info(
                    f"({len(results)}/{len(grid)}): Finished grid point {res[0]} with solution {res[1]}"
                )
                results.append(res)
            except Exception as e:
                logger.warning(f"({len(results)}/{len(grid)}): An error occurred: {e}")
            finally:
                active_offsets.remove(future.thread_offset)

    return results


if __name__ == "__main__":
    args = parse_args()

    sweep_runner(
        args=args,
    )
