import sys, pdb, traceback
import time


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# Set the custom exception hook
sys.excepthook = excepthook

import torch
from models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from models.gpt2.configuration_gpt2 import GPT2Config

cf = GPT2Config.from_pretrained("/data/huggingface/nice-gpt2-1.5b")
cf.activation_function = "gelu"
cf._attn_implementation = "sdpa"

model = GPT2LMHeadModel(cf).to("cuda:0")

elapsed_times = []
for itr in range(10):
    print(f"Running iter: {itr}")
    input_tensor = torch.randint(
        1,
        10000,
        (8, 128),
    ).to("cuda:0")
    start_time = time.time()
    out = model(input_tensor)
    end_time = time.time()
    elapsed_times.append(end_time - start_time)
    print(f"Time taken: {end_time - start_time}s")
    print(f"Out device: {out['logits'].device}")

print(f"Average time: {sum(elapsed_times[2:]) / len(elapsed_times[2:])}")
