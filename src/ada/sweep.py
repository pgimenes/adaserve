from copy import copy
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path
import re

import numpy as np

from chop.tools import get_logger

from ada.auto import autosharding_runner
from ada.plot import plot_bs_seq_len

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def autosharding_runner_wrapper(model_class=None, model_config=None, args=None):
    """
    Pick the outputs we want from the autosharding_analysis_pass
    """
    _, out = autosharding_runner(model_class, model_config, args)
    return (
        (args.batch_size, args.sequence_length),
        out["autosharding_analysis_pass"]["solution"],
    )


def evaluate_grid_point(args):
    bs, seq_len = args.grid_point
    log_filename = f"experiments/sweep/sweep_{args.model}_layers_{args.num_hidden_layers}_bs_{bs}_seq_len_{seq_len}.log"
    command = [
        "python",
        "src/main.py",
        "--auto",
        "--skip-forward",
        # Model config
        "--model",
        str(args.model),
        "--num_hidden_layers",
        str(args.num_hidden_layers),
        "--batch_size",
        str(bs),
        "--sequence_length",
        str(seq_len),
        # Optimizer args
        "--optimizer_profile",
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

        # Extract solution from logfile with regex to match
        # Autosharding pass complete. Time taken: 10.546476125717163 seconds. Solution: 4001085.363838052
        with open(log_filename, "r") as file:
            for line in file:
                if "Autosharding pass complete" in line:
                    pattern = r"[-+]?\d*\.\d+|\d+"
                    floats = [i for i in re.findall(pattern, line) if "." in i]
                    solution = floats[-1]
                    return ((bs, seq_len), float(solution))

    except subprocess.CalledProcessError as e:
        logger.warning(f"Grid point {bs} failed with error: {e}")
        return None


def sweep_runner(model_class=None, model_config=None, args=None):
    # Create experiments output directory if it doesn't exist
    Path("experiments/sweep").mkdir(parents=True, exist_ok=True)

    bs_range = np.linspace(
        start=args.sweep_min_bs, stop=args.sweep_max_bs, num=args.sweep_grid_size
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
        for grid_point in grid:
            nargs = copy(args)
            setattr(nargs, "grid_point", grid_point)
            futures.append(executor.submit(evaluate_grid_point, nargs))

        # Process trial outputs
        results = []
        for future in as_completed(futures):
            try:
                res = future.result()
                logger.debug(
                    f"({len(results)}/{len(grid)}): Finished grid point {res[0]} with solution {res[1]}"
                )
                results.append(res)
            except Exception as e:
                logger.warning(f"An error occurred: {e}")

    # Save results to csv file
    with open("experiments/sweep/sweep_results.csv", "w") as file:
        for result in results:
            file.write(f"{result[0][0]},{result[0][1]},{result[1]}\n")

    # Make bs/slen/solution plot
    plot_bs_seq_len(results)
