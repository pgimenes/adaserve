from copy import copy
import itertools
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
    ]

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


def sweep_runner(
    args=None,
):
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

        # Launch trials
        futures = []
        for idx, grid_point in enumerate(grid):
            thread_offset = idx // args.sweep_max_threads
            nargs = copy(args)
            setattr(nargs, "grid_point", grid_point)
            futures.append(executor.submit(evaluate_grid_point, nargs))

        # Process trial outputs
        results = []
        for future in as_completed(futures):
            try:
                res = future.result()
                logger.info(
                    f"({len(results)}/{len(grid)}): Finished grid point {res[0]} with solution {res[1]}"
                )
                results.append(res)
            except Exception as e:
                logger.warning(f"({len(results)}/{len(grid)}): An error occured: {e}")


if __name__ == "__main__":
    args = parse_args()

    sweep_runner(
        args=args,
    )
