import sys, pdb, traceback


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# Set the custom exception hook
sys.excepthook = excepthook

import torch
from models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from models.gpt2.configuration_gpt2 import GPT2Config

cf = GPT2Config()
cf.activation_function = "gelu"

model = GPT2LMHeadModel(cf)

_ = model(
    torch.randint(
        1,
        10000,
        (1, 128),
    ),
)
