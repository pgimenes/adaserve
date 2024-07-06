import torch

from chop import AutoPipelineForDistributedInference
from chop.ir import MaseGraph
from chop.tools import get_logger

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

    mg = pipeline(
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
            },
            "resharding_transform_pass": {
                "module_map": "self/autosharding_analysis_pass",  # output of autosharding_analysis_pass is directed to resharding_transform_pass
                "device_mesh": args.device_mesh,
            },
        },
    )

    mg.draw()
