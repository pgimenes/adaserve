import sys, pdb, traceback
import argparse

from chop.tools import get_logger
from chop.distributed.launcher import MaseLauncher

from models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from models.gpt2.configuration_gpt2 import GPT2Config

from manual import manual_sharding_runner
from auto import autosharding_runner
from sweep import sweep_runner
from distributed import device_fn

logger = get_logger(__name__)
logger.setLevel("INFO")

CONFIG_MAP = {
    "gpt2": GPT2Config,
}

MODEL_MAP = {
    "gpt2": GPT2LMHeadModel,
}


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


sys.excepthook = excepthook


def parse_args():
    parser = argparse.ArgumentParser()

    # Action (running mode)
    parser.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Run autosharding pass on defined model.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a sweep over input tensor profiles.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual sharding for testing/debugging. If not selected, will run autosharding instead.",
    )
    parser.add_argument(
        "--pre_mp_autosharding",
        action="store_true",
        help="Run the autosharding pass before spawning multiprocessing for debugging.",
    )

    # Autosharding args
    parser.add_argument(
        "--algo",
        type=str,
        default="alpa",
        choices=["alpa", "megatron"],
        help="Algorithm to use for sharding the model.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload solution from file. Expected location: experiments/<model>_bs_<batch_size>_seq_len_<sequence_length>_milp_gap_<optimizer_mip_rel_gap>_ilp_solution.pkl",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Run autosharding pass to extract optimal sharding configuration but skip forward pass.",
    )

    # Manual sharding args
    parser.add_argument(
        "--row", action="store_true", help="Use row sharding (manual) for testing."
    )
    parser.add_argument(
        "--column",
        action="store_true",
        help="Use column sharding (manual) for testing.",
    )

    # Additional options
    parser.add_argument(
        "--optimizer_profile",
        action="store_true",
        help="Output profiling for the ILP optimizer.",
    )
    parser.add_argument(
        "--optimizer_time_limit",
        type=int,
        default=10000,
        help="The maximum number of seconds allotted to solve the problem.",
    )
    parser.add_argument(
        "--optimizer_mip_rel_gap",
        type=int,
        default=98,
        help="Termination criterion for MIP solver: terminal when primal-dual gap <= mip_rel_gap.",
    )

    # Define model
    parser.add_argument(
        "--model",
        choices=["gpt2"],
        default="gpt2",
        help="Specify the model to use (gpt2)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Specify the Huggingface checkpoint",
    )
    parser.add_argument(
        "--from_config",
        action="store_true",
        help="Instantiate new model from config. If not set, will load from checkpoint.",
    )

    # Huggingface config options
    parser.add_argument(
        "--_attn_implementation",
        type=str,
        default=None,
        help="Attention implementation",
    )

    # OPT
    parser.add_argument(
        "--ffn_dim", type=int, default=None, help="Number of hidden layers"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=None,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--word_embed_proj_dim",
        type=int,
        default=None,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--activation_function",
        type=str,
        default=None,
        help="Activation function'",
    )

    # Bert
    parser.add_argument(
        "--intermediate_size", type=int, default=None, help="Intermediate size"
    )

    # Other configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Intermediate size",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Intermediate size",
    )

    # Environment setup
    parser.add_argument(
        "--world_size",
        type=int,
        default=8,
        help="Number of GPU devices.",
    )
    parser.add_argument(
        "--device_mesh",
        type=list,
        default=[[0, 1, 2, 3], [4, 5, 6, 7]],
        help="2D mesh of device IDs.",
    )
    parser.add_argument(
        "--mesh_shape",
        type=tuple,
        default=(2, 4),
        help="Shape of logical device mesh.",
    )

    # Sweep parameters (only relevant if sweep is selected)
    parser.add_argument(
        "--sweep-max-threads",
        type=int,
        default=16,
        help="Max number of threads for sweep.",
    )
    parser.add_argument(
        "--sweep-grid-size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--sweep-min-bs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--sweep-max-bs",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--sweep-min-seq-len",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--sweep-max-seq-len",
        type=int,
        default=1000,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Define config and model
    if args.checkpoint is None:
        config = CONFIG_MAP[args.model]()
    else:
        config = CONFIG_MAP[args.model].from_pretrained(args.checkpoint)

    model_class = MODEL_MAP[args.model]

    # Update config parameters according to CLI arguments
    for arg in [
        "ffn_dim",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "word_embed_proj_dim",
        "intermediate_size",
        "_attn_implementation",
        "activation_function",
    ]:
        cli_arg = getattr(args, arg, None)
        if cli_arg is not None:
            logger.debug(f"Setting {arg} to {cli_arg}")
            setattr(config, arg, cli_arg)

    # Run manual sharding if requested
    if args.manual:
        logger.info(f"Running manual sharding for model: {args.model}")
        manual_sharding_runner(model_class=model_class, model_config=config, args=args)

    # Sweep autosharding for several input request profiles
    elif args.sweep:
        sweep_runner(model_class=model_class, model_config=config, args=args)

    # Run autosharding if requested
    else:
        logger.info(f"Running autosharding for model: {args.model}")

        if args.pre_mp_autosharding:
            mg, pass_outputs = autosharding_runner(
                model_class=model_class, model_config=config, args=args
            )
            return

        if not args.skip_forward:
            # Launch model in distributed cluster
            launcher = MaseLauncher(
                world_size=args.world_size,
                device_mesh=args.device_mesh,
                device_fn=device_fn,
            )

            launcher.run(model_class, config, args)


if __name__ == "__main__":
    main()
