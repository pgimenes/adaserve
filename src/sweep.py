import multiprocessing
from copy import copy
import itertools

import numpy as np
import matplotlib.pyplot as plt

from chop import AutoPipelineForDistributedInference
from chop.ir import MaseGraph
from chop.tools import get_logger
from chop.distributed import MaseLauncher

from auto import autosharding_runner

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


def plot(results):
    # Extract the values from results
    bs = [result[0][0] for result in results]
    seq_len = [result[0][1] for result in results]
    autosharding_time = [result[1] for result in results]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the data
    ax.scatter(bs, seq_len, autosharding_time)

    # Set labels
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Sequence Length")
    ax.set_zlabel("Autosharding Time")

    # Show the plot
    plt.savefig("out.png")
    plt.show()


def sweep_runner(model_class=None, model_config=None, args=None):
    bs_range = np.linspace(
        start=args.min_bs, stop=args.max_bs, num=args.num_evals
    ).astype(int)

    seq_len_range = np.linspace(
        start=args.min_seq_len, stop=args.max_seq_len, num=args.num_evals
    ).astype(int)

    results = []
    grid = [i for i in itertools.product(bs_range, seq_len_range)]

    for idx, grid_point in enumerate(grid):
        bs, sl = grid_point

        new_args = copy(args)
        new_args.batch_size = bs
        new_args.sequence_length = sl

        logger.info(f"Sweep iteration {idx}/{len(grid)}: {new_args}")
        result = autosharding_runner_wrapper(model_class, model_config, new_args)
        results.append(result)

    # with multiprocessing.Pool(processes=args.sweep_max_threads) as pool:
    #     results = pool.starmap(autosharding_runner_wrapper, arg_list)

    print(results)
    plot(results)
