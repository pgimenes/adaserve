import torch
import torch.nn as nn

from chop.ir import MaseGraph
from chop.distributed import MaseLauncher
import chop.passes as passes
from chop.tools import get_logger
from chop.pipelines import AutoPipelineForDistributedInference

from models.bert.modeling_bert import BertModel
from models.bert.configuration_bert import BertConfig

logger = get_logger(__name__)
logger.setLevel("DEBUG")

WORLD_SIZE = 8
DEVICE_MESH = [[0, 1, 2, 3], [4, 5, 6, 7]]

import sys, pdb, traceback


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# Set the custom exception hook
sys.excepthook = excepthook


def test_autosharding():

    # Define config
    config = BertConfig()
    config.num_hidden_layers = 1
    config.hidden_size = 96
    config.intermediate_size = 384
    config._attn_implementation = "eager"
    config_batch_size = 5
    # config_batch_size = 10 * 1024
    config_sequence_length = 96

    # Initialize model and MaseGraph
    model = BertModel(config)
    mg = MaseGraph(model)

    pipe = AutoPipelineForDistributedInference()

    mg = pipe(
        mg,
        pass_args={
            "report_graph_analysis_pass": {"file_name": "bert.txt"},
            "add_common_metadata_analysis_pass": {
                "dummy_in": {
                    "input_ids": torch.randn(
                        (config_batch_size, config_sequence_length, config.hidden_size)
                    ),
                },
                "add_value": True,
            },
            "autosharding_analysis_pass": {
                "mesh_shape": (2, 4),
                "inter_node_bandwidth": 10e9,
                "intra_node_bandwidth": 100e9,
                "skip_fully_replicated": True,
            },
            "resharding_transform_pass": {
                "device_mesh": DEVICE_MESH,
                # Take output from autosharding_analysis_pass
                "module_map": "self/autosharding_analysis_pass",
            },
        },
    )

    # Launch model in distributed cluster
    # launcher = MaseLauncher(mg, world_size=WORLD_SIZE, device_mesh=DEVICE_MESH)
    # inputs = [torch.randint(0, 10, (1, config_sequence_length))]
    # inputs = [torch.randn((config_batch_size, config_sequence_length, config.hidden_size))]
    # launcher.run(module_map, inputs)


if __name__ == "__main__":
    test_autosharding()
