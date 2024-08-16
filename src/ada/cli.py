import argparse

from ada.models import CONFIG_MAP, MODEL_MAP


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
    autoshard_group.add_argument(
        "--benchmarking_device",
        type=int,
        default=0,
        help="Which GPU device to use for profiling compute costs for the ILP objective function.",
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
        default=0,
        help="Termination criterion for MIP solver: terminal when primal-dual gap <= mip_rel_gap.",
    )

    # Sweep configuration
    # --------------------------------------------
    sweep_group = parser.add_argument_group("Sweep configuration")
    sweep_group.add_argument(
        "--sweep-max-threads",
        type=int,
        default=8,
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
        default=1,
    )
    sweep_group.add_argument(
        "--sweep-max-bs",
        type=int,
        default=32,
    )
    sweep_group.add_argument(
        "--sweep-min-seq-len",
        type=int,
        default=128,
    )
    sweep_group.add_argument(
        "--sweep-max-seq-len",
        type=int,
        default=2**14,
    )

    args = parser.parse_args()
    return args


def get_model_from_args(args):
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
            setattr(config, arg, cli_arg)

    return model_class, config
