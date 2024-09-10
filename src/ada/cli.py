import argparse

def get_cli_args():
    parser = argparse.ArgumentParser()

    # Action (running mode)
    # --------------------------------------------
    action_group = parser.add_argument_group("Action (running mode)")
    action_group.add_argument(
        "--dynamic_resharding",
        action="store_true",
        help="Enable dynamic reshading for the model",
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
        "--tensor_parallel",
        type=int,
        default=1,
        help="Number of GPU devices.",
    )

    # Model configuration
    # --------------------------------------------
    model_group = parser.add_argument_group("Model configuration")
    model_group.add_argument(
        "--model_name",
        default=None,
        help="Specify the Huggingface checkpoint",
    )
    model_group.add_argument(
        "--from_pretrained",
        action="store_true",
        help="Load pretrained weights from checkpoint",
    )
    model_group.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    model_group.add_argument(
        "--datatype",
        type=str,
        default="float32",
    )

    # Dataset configuration
    # --------------------------------------------
    dataset_group = parser.add_argument_group("Dataset configuration")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="azure_conv",
    )
    dataset_group.add_argument(
        "--max_requests",
        type=int,
        default=None,
        help="Maximum number of requests to process in the dataset",
    )

    # Output configuration
    # --------------------------------------------

    output_group = parser.add_argument_group("Output configuration")
    output_group.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output file.",
    )

    args = parser.parse_args()
    return args