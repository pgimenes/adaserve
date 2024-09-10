import sys, pdb, traceback


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# Set the custom exception hook
sys.excepthook = excepthook

import torch
import torch.nn as nn


from ada.nn.manual import ManualBatchLinear

vec = torch.randn(50)
mat = torch.randn((10, 50))
bmat = torch.randn((10, 10, 50))
large_bmat = torch.randn((10, 10, 10, 50))

layer_no_bias = ManualBatchLinear(50, 100, bias=False)
layer_bias = ManualBatchLinear(50, 100, bias=True)

# out = layer_no_bias(vec)
# print(out.shape)
# out = layer_bias(vec)
# print(out.shape)

out = layer_no_bias(mat)
print(out.shape)
out = layer_bias(mat)
print(out.shape)

out = layer_no_bias(bmat)
print(out.shape)
out = layer_bias(bmat)
print(out.shape)

out = layer_no_bias(large_bmat)
print(out.shape)
out = layer_bias(large_bmat)
print(out.shape)
