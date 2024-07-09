import argparse
import sys

from chop.tools import get_logger

from models.toy.configuration_toy import ToyConfig
from models.toy.toy_model import ToyModel
from models.bert.configuration_bert import BertConfig
from models.bert.modeling_bert import BertModel
from models.gpt2.modeling_gpt2 import GPT2Model
from models.gpt2.configuration_gpt2 import GPT2Config

from manual import manual_sharding_runner
from auto import autosharding_runner
from sweep import sweep_runner

logger = get_logger(__name__)
logger.setLevel("DEBUG")

CONFIG_MAP = {"toy": ToyConfig, "bert": BertConfig, "gpt2": GPT2Config}
MODEL_MAP = {"toy": ToyModel, "bert": BertModel, "gpt2": GPT2Model}


def parse_args():
    parser = argparse.ArgumentParser()

    # Action
    parser.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Run a sweep over input tensor profiles.",
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

    # Manual sharding
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
        default=0,
        help="Termination criterion for MIP solver: terminal when primal-dual gap <= mip_rel_gap.",
    )

    # Define model
    parser.add_argument(
        "--model",
        choices=["bert", "gpt2", "toy"],
        default="toy",
        help="Specify the model to use (toy/bert/gpt2)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Specify the Huggingface checkpoint",
    )

    # Huggingface config options
    parser.add_argument(
        "--num_hidden_layers", type=int, default=None, help="Number of hidden layers"
    )
    parser.add_argument("--hidden_size", type=int, default=None, help="Hidden size")
    parser.add_argument(
        "--intermediate_size", type=int, default=None, help="Intermediate size"
    )
    parser.add_argument(
        "--_attn_implementation",
        type=str,
        default=None,
        help="Attention implementation",
    )

    # Other configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Intermediate size")
    parser.add_argument(
        "--sequence_length", type=int, default=128, help="Intermediate size"
    )

    # Environment setup
    parser.add_argument(
        "--world_size", type=int, default=8, help="Number of GPU devices."
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
    parser.add_argument("--sweep-grid-size", type=int, default=10)
    parser.add_argument("--sweep-min-bs", type=int, default=10)
    parser.add_argument("--sweep-max-bs", type=int, default=1000)
    parser.add_argument("--sweep-min-seq-len", type=int, default=10)
    parser.add_argument("--sweep-max-seq-len", type=int, default=1000)

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
        "hidden_size",
        "intermediate_size",
        "_attn_implementation",
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
        mg, pass_outputs = autosharding_runner(
            model_class=model_class, model_config=config, args=args
        )


if __name__ == "__main__":
    main()
