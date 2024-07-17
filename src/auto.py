import torch

from chop import AutoPipelineForDistributedInference
from chop.ir import MaseGraph
from chop.tools import get_logger
from chop.distributed import MaseLauncher
import chop.passes as passes

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def autosharding_runner(model_class=None, model_config=None, args=None):

    model = model_class(model_config)

    mg = MaseGraph(
        model,
        # Don't include embedding nodes in graph
        hf_input_names=["inputs_embeds"],
    )
    pipeline = AutoPipelineForDistributedInference()

    model_name = (
        args.checkpoint.replace("/", "-").replace(".", "-")
        if args.checkpoint is not None
        else args.model
    )

    # Skip embedding layer
    inputs = torch.randn(
        (args.batch_size, args.sequence_length, model_config.hidden_size)
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
                    "inputs_embeds": torch.randn((1, 128, model_config.hidden_size)),
                },
                "add_value": True,
            },
            "autosharding_analysis_pass": {
                "mesh_shape": args.mesh_shape,
                "inter_node_bandwidth": 10e9,
                "intra_node_bandwidth": 100e9,
                "skip_fully_replicated": True,
                "time_limit": args.optimizer_time_limit,
                "mip_rel_gap": args.optimizer_mip_rel_gap,
                "run_checks": False,
                "preload_solution": args.preload,
                "ilp_solution_file": f"experiments/{model_name}_full_solution_bs_{args.batch_size}_seq_len_{args.sequence_length}_milp_gap_{args.optimizer_mip_rel_gap}_ilp_solution.pkl",
            },
            "resharding_transform_pass": {
                "tensor_sharding_map": "self/autosharding_analysis_pass",  # output of autosharding_analysis_pass is directed to resharding_transform_pass
                "device_mesh": args.device_mesh,
            },
        },
        skip_passes=[
            passes.resharding_transform_pass,
            passes.graph.analysis.report.report_parallelization_analysis_pass,
        ],
    )

    # Skip drawing for larger graphs to reduce runtime
    if args.num_hidden_layers == 1:
        mg.draw()

    if not args.skip_forward:
        # Launch model in distributed cluster
        launcher = MaseLauncher(
            mg, world_size=args.world_size, device_mesh=args.device_mesh
        )
        launcher.run(
            pipeline.pass_outputs["autosharding_analysis_pass"]["tensor_sharding_map"],
            inputs=[inputs],
        )

    return mg, pass_outputs
