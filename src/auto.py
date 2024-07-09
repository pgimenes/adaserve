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

    mg = MaseGraph(model)
    pipeline = AutoPipelineForDistributedInference()

    report_graph_fname = (
        f"{args.checkpoint.replace('/', '-')}-graph.txt"
        if args.checkpoint is not None
        else f"{args.model}-graph.txt"
    )

    # Skip embeddings for bert
    if args.model in ["toy", "bert"]:
        input_ids = torch.randn(
            (args.batch_size, args.sequence_length, model_config.hidden_size)
        )
    else:
        input_ids = torch.randint(0, 10, (args.batch_size, args.sequence_length))

    mg, pass_outputs = pipeline(
        mg,
        pass_args={
            "report_graph_analysis_pass": {"file_name": report_graph_fname},
            "add_common_metadata_analysis_pass": {
                "dummy_in": {
                    "input_ids": input_ids,
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
                "run_checks": True,
            },
            "resharding_transform_pass": {
                "tensor_sharding_map": "self/autosharding_analysis_pass",  # output of autosharding_analysis_pass is directed to resharding_transform_pass
                "device_mesh": args.device_mesh,
            },
        },
        skip_passes=[passes.resharding_transform_pass],
    )

    mg.draw()

    # Launch model in distributed cluster
    launcher = MaseLauncher(
        mg, world_size=args.world_size, device_mesh=args.device_mesh
    )
    launcher.run(
        pipeline.pass_outputs["autosharding_analysis_pass"]["tensor_sharding_map"],
        inputs=[input_ids],
    )

    return mg, pass_outputs
