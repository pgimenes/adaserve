from .gpt2.modeling_gpt2 import GPT2LMHeadModel
from .gpt2.configuration_gpt2 import GPT2Config

CONFIG_MAP = {
    "gpt2": GPT2Config,
}

MODEL_MAP = {
    "gpt2": GPT2LMHeadModel,
}
