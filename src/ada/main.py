import os

from chop.tools import get_logger
from chop.distributed.launcher import MaseLauncher

from ada.cli import parse_args, get_model_from_args
from ada.manual import manual_sharding_runner
from ada.models import CONFIG_MAP, MODEL_MAP
from ada.auto import autosharding_runner
from ada.distributed import single_batch_device_fn, serving_device_fn

logger = get_logger(__name__)
logger.setLevel("INFO")


def main():
    args = parse_args()

    model_class, config = get_model_from_args(args)

    # Run manual sharding if requested
    if args.row or args.column:
        logger.info(f"Running manual sharding for model: {args.model}")
        manual_sharding_runner(
            model_class=model_class,
            model_config=config,
            args=args,
        )

    # Run autosharding if requested
    else:
        logger.info(f"Running autosharding for model: {args.model}")

        if args.skip_forward:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.benchmarking_device)

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
