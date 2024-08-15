import argparse

from chop.tools import get_logger
from chop.distributed.launcher import MaseLauncher

from ada.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from ada.models.gpt2.configuration_gpt2 import GPT2Config

from ada.manual import manual_sharding_runner
from ada.auto import autosharding_runner
from ada.sweep import sweep_runner
from ada.distributed import single_batch_device_fn, serving_device_fn

logger = get_logger(__name__)
logger.setLevel("INFO")

CONFIG_MAP = {
    "gpt2": GPT2Config,
}

MODEL_MAP = {
    "gpt2": GPT2LMHeadModel,
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Action (running mode)
    # --------------------------------------------
    action_group = parser.add_argument_group("Action (running mode)")
    action_group.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Run autosharding on a single design point, defined by batch size and sequence length.",
    )
    action_group.add_argument(
        "--row",
        action="store_true",
        help="Use row sharding (manual) for testing. Default: false",
    )
    action_group.add_argument(
        "--column",
        action="store_true",
        help="Use column sharding (manual) for testing. Default: false",
    )
    action_group.add_argument(
        "--sweep",
        action="store_true",
        help="Run a sweep over input tensor profiles to collect autosharding solutions. Default: false",
    )
    action_group.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode, with extra verbosity and checks.",
    )

    # Environment setup
    # --------------------------------------------
    env_group = parser.add_argument_group("Environment setup")
    env_group.add_argument(
        "--world_size",
        type=int,
        default=8,
        help="Number of GPU devices.",
    )
    env_group.add_argument(
        "--inter_node_bandwidth",
        type=int,
        default=10e9,
        help="Number of GPU devices.",
    )
    env_group.add_argument(
        "--intra_node_bandwidth",
        type=int,
        default=100e9,
        help="Number of GPU devices.",
    )

    # Model configuration
    # --------------------------------------------
    model_group = parser.add_argument_group("Model configuration")
    model_group.add_argument(
        "--model",
        choices=["gpt2"],
        default=None,
        required=True,
        help="Specify the model to use (gpt2)",
    )
    model_group.add_argument(
        "--checkpoint",
        default=None,
        help="Specify the Huggingface checkpoint",
    )
    model_group.add_argument(
        "--from_pretrained",
        action="store_true",
        help="Load pretrained weights from checkpoint",
    )

    # Huggingface config options (override checkpoint)
    model_group.add_argument(
        "--_attn_implementation",
        type=str,
        default=None,
        help="Attention implementation",
    )
    model_group.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="Number of hidden layers",
    )

    # Dataset configuration
    # --------------------------------------------
    dataset_group = parser.add_argument_group("Dataset configuration")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to use for benchmarking.",
    )

    # Autosharding configuration
    # --------------------------------------------
    autoshard_group = parser.add_argument_group("Autosharding configuration")
    autoshard_group.add_argument(
        "--algo",
        type=str,
        default="alpa",
        choices=[
            "alpa",
            "megatron",
            "fully_replicated",
        ],
        help="Algorithm to use for sharding the model.",
    )
    autoshard_group.add_argument(
        "--preload",
        action="store_true",
        help="Preload solution from file. Expected location: experiments/<model>_bs_<batch_size>_seq_len_<sequence_length>_milp_gap_<optimizer_mip_rel_gap>_ilp_solution.pkl",
    )
    autoshard_group.add_argument(
        "--skip_forward",
        action="store_true",
        help="Run autosharding pass to extract optimal sharding configuration but skip forward pass.",
    )

    # Design point
    autoshard_group.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Intermediate size",
    )
    autoshard_group.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Intermediate size",
    )

    # ILP optimizer configuration
    # --------------------------------------------
    ilp_group = parser.add_argument_group("ILP optimizer configuration")
    ilp_group.add_argument(
        "--optimizer_profile",
        action="store_true",
        help="Output profiling for the ILP optimizer.",
    )
    ilp_group.add_argument(
        "--optimizer_time_limit",
        type=int,
        default=10000,
        help="The maximum number of seconds allotted to solve the problem.",
    )
    ilp_group.add_argument(
        "--optimizer_mip_rel_gap",
        type=int,
        default=98,
        help="Termination criterion for MIP solver: terminal when primal-dual gap <= mip_rel_gap.",
    )

    # Sweep configuration
    # --------------------------------------------
    sweep_group = parser.add_argument_group("Sweep configuration")
    sweep_group.add_argument(
        "--sweep-max-threads",
        type=int,
        default=16,
        help="Max number of threads for sweep.",
    )
    sweep_group.add_argument(
        "--sweep-grid-size",
        type=int,
        default=10,
    )
    sweep_group.add_argument(
        "--sweep-min-bs",
        type=int,
        default=10,
    )
    sweep_group.add_argument(
        "--sweep-max-bs",
        type=int,
        default=1000,
    )
    sweep_group.add_argument(
        "--sweep-min-seq-len",
        type=int,
        default=10,
    )
    sweep_group.add_argument(
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
        "num_hidden_layers",
        "_attn_implementation",
    ]:
        cli_arg = getattr(args, arg, None)
        if cli_arg is not None:
            logger.debug(f"Setting {arg} to {cli_arg}")
            setattr(config, arg, cli_arg)

    # Run manual sharding if requested
    if args.row or args.column:
        logger.info(f"Running manual sharding for model: {args.model}")
        manual_sharding_runner(
            model_class=model_class,
            model_config=config,
            args=args,
        )

    # Sweep autosharding for several input request profiles
    elif args.sweep:
        sweep_runner(
            model_class=model_class,
            model_config=config,
            cli_args=args,
        )

    # Run autosharding if requested
    else:
        logger.info(f"Running autosharding for model: {args.model}")

        if args.skip_forward:
            mg, pass_outputs = autosharding_runner(
                model_class=model_class,
                model_config=config,
                cli_args=args,
            )
            return

        else:
            # Launch model in distributed cluster
            spmd_fn = (
                single_batch_device_fn if args.dataset is None else serving_device_fn
            )

            launcher = MaseLauncher(
                world_size=args.world_size,
                device_fn=spmd_fn,
            )

            launcher.run(model_class, config, args)


if __name__ == "__main__":
    main()
