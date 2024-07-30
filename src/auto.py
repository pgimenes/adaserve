import torch

from chop import AutoPipelineForDistributedInference
from chop.ir import MaseGraph
from chop.tools import get_logger
import chop.passes as passes

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def autosharding_runner(
    model_class=None,
    model_config=None,
    args=None,
    inputs=None,
):

    cli_args = args

    if cli_args.from_config:
        model = model_class(model_config)
    else:
        model = model_class.from_pretrained(cli_args.checkpoint)

    mg = MaseGraph(
        model,
        # Don't include embedding nodes in graph
        hf_input_names=["inputs_embeds"],
    )
    pipeline = AutoPipelineForDistributedInference()

    model_name = (
        cli_args.checkpoint.replace("/", "-").replace(".", "-")
        if cli_args.checkpoint is not None
        else cli_args.model
    )

    model_name = model_name[1:] if model_name.startswith("-") else model_name

    # Skip embedding layer
    if inputs is None:
        inputs = torch.randn(
            (cli_args.batch_size, cli_args.sequence_length, model_config.hidden_size)
        )

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
                "mesh_shape": cli_args.mesh_shape,
                "inter_node_bandwidth": 10e9,
                "intra_node_bandwidth": 100e9,
                "skip_fully_replicated": True,
                "time_limit": cli_args.optimizer_time_limit,
                "mip_rel_gap": cli_args.optimizer_mip_rel_gap,
                "run_checks": False,
                "preload_solution": cli_args.preload,
                "ilp_solution_file": f"experiments/{model_name}_bs_{cli_args.batch_size}_seq_len_{cli_args.sequence_length}_milp_gap_{cli_args.optimizer_mip_rel_gap}_ilp_solution.pkl",
            },
            "resharding_transform_pass": {
                "tensor_sharding_map": "self/autosharding_analysis_pass",  # output of autosharding_analysis_pass is directed to resharding_transform_pass
                "device_mesh": cli_args.device_mesh,
            },
        },
        skip_passes=[
            # passes.resharding_transform_pass,
            passes.graph.analysis.report.report_parallelization_analysis_pass,
        ],
    )

    # Skip drawing for larger graphs to reduce runtime
    if args.num_hidden_layers == 1:
        mg.draw()

    return mg, pass_outputs
