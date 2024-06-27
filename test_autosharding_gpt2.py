import torch

from chop import AutoPipelineForDistributedInference
from chop.ir import MaseGraph
from chop.distributed import MaseLauncher
from chop.tools import get_logger

from models.gpt2.modeling_gpt2 import GPT2Model
from models.gpt2.configuration_gpt2 import GPT2Config

logger = get_logger(__name__)
logger.setLevel("DEBUG")

WORLD_SIZE = 2
DEVICE_MESH = [[0, 1]]
MESH_SHAPE = (1, 2)

def test_autosharding():
    config_sequence_length = 128
    config_batch_size = 10 * 1024

    # Initialize model and MaseGraph
    checkpoint = "openai-community/gpt2"
    # model = GPT2Model.from_pretrained(checkpoint)
    config = GPT2Config()
    config.n_layer = 1
    model = GPT2Model(config)

    from transformers.pytorch_utils import Conv1D
    CUSTOM_MODULES = {
        "modules": {
            Conv1D: {
                "args": {
                    "x": "data_in"
                }
            },
        },
        "functions": {}
    }

    mg = MaseGraph(model, custom_ops=CUSTOM_MODULES)
    pipeline = AutoPipelineForDistributedInference()

    mg = pipeline(
        mg, 
        pass_args={
            "report_graph_analysis_pass": {
                "file_name": f"{checkpoint.replace('/', '-')}-graph.txt"
            },
            "add_common_metadata_analysis_pass": {
                "dummy_in": {
                    "input_ids": torch.randint(0, 10, (1, config_sequence_length)),
                    # "input_ids": torch.randn((1, config_sequence_length, config.hidden_size)),
                },
                "add_value": False,
            },
            "autosharding_analysis_pass": {
                "mesh_shape": MESH_SHAPE,
                "inter_node_bandwidth": 10e9,
                "intra_node_bandwidth": 100e9
            },
            "resharding_transform_pass": {
                "module_map": "self/autosharding_analysis_pass", # output of autosharding_analysis_pass is directed to resharding_transform_pass
                "device_mesh": DEVICE_MESH
            }
        }
    )

    mg.draw()

    # Launch model in distributed cluster
    inputs = [torch.randint(0, 10, (config_batch_size, config_sequence_length))]
    # inputs = [torch.randn((config_batch_size, config_sequence_length, config.hidden_size))]
    # launcher = MaseLauncher(mg, world_size=WORLD_SIZE, device_mesh=DEVICE_MESH)
    # launcher.run(pipeline.pass_outputs["autosharding_analysis_pass"], inputs)


if __name__ == "__main__":
    test_autosharding()
