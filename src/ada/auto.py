import torch
import numpy as np

from chop import AutoPipelineForDistributedInference
from chop.ir import MaseGraph
from chop.tools import get_logger
import chop.passes as passes
from chop.distributed.utils import _get_mesh_from_world_size

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def get_cached_solution_fname(model_name, cli_args):
    solution_fname = f"experiments/{model_name}_bs_{cli_args.batch_size}_seq_len_{cli_args.sequence_length}_milp_gap_{cli_args.optimizer_mip_rel_gap}_"

    # If any non-default configuration parameters are passed,
    # add to the solution name
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
        cli_arg = getattr(cli_args, arg, None)
        if cli_arg is not None:
            solution_fname += f"{arg}_{cli_arg}_"

    solution_fname += "ilp_solution.pkl"

    return solution_fname


def autosharding_runner(
    model_class=None,
    model_config=None,
    cli_args=None,
    inputs=None,
):

    # Get model
    if cli_args.from_pretrained:
        model = model_class.from_pretrained(cli_args.checkpoint)
    else:
        model = model_class(model_config)

    model_name = (
        cli_args.checkpoint.replace("/", "-").replace(".", "-")
        if cli_args.checkpoint is not None
        else cli_args.model
    )

    model_name = model_name[1:] if model_name.startswith("-") else model_name

    # Get mesh info
    mesh_ids, mesh_shape = _get_mesh_from_world_size(cli_args.world_size)

    # Generate MaseGraph
    mg = MaseGraph(
        model,
        # Don't include embedding nodes in graph
        hf_input_names=["inputs_embeds"],
    )
    pipeline = AutoPipelineForDistributedInference()

    # Get dummy inputs
    if inputs is None:
        inputs = torch.randn(
            (
                cli_args.batch_size,
                cli_args.sequence_length,
                model_config.hidden_size,
            )
        )

    # Run pipeline
    mg, pass_outputs = pipeline(
        mg,
        pass_args={
            "report_graph_analysis_pass": {
                "file_name": f"{model_name}-graph.txt",
            },
            "add_common_metadata_analysis_pass": {
                # TO DO: change key according to model (non-HuggingFace)
                "dummy_in": {
                    "inputs_embeds": inputs,
                },
                "add_value": True,
            },
            "autosharding_analysis_pass": {
                "algo": cli_args.algo,
                "mesh_shape": mesh_shape,
                "inter_node_bandwidth": cli_args.inter_node_bandwidth,
                "intra_node_bandwidth": cli_args.intra_node_bandwidth,
                "skip_fully_replicated": True,
                "time_limit": cli_args.optimizer_time_limit,
                "mip_rel_gap": cli_args.optimizer_mip_rel_gap,
                "run_checks": cli_args.debug,
                "preload_solution": cli_args.preload,
                "ilp_solution_file": get_cached_solution_fname(model_name, cli_args),
                "benchmarking_device": cli_args.benchmarking_device,
            },
            "resharding_transform_pass": {
                "tensor_sharding_map": "self/autosharding_analysis_pass",  # output of autosharding_analysis_pass is directed to resharding_transform_pass
                "device_mesh": mesh_ids,
            },
        },
    )

    # Skip drawing for larger graphs to reduce runtime
    mg.draw()

    return mg, pass_outputs
