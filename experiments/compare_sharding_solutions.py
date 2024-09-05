import dill
from tabulate import tabulate

value1 = 1152

sharding_config = {}
for layer in range(96):
    sharding_config[f"transformer.h.{layer}.ln_1"] = "replicated"
    sharding_config[f"transformer.h.{layer}.attn.c_attn"] = "column"
    sharding_config[f"transformer.h.{layer}.attn.attn"] = "head"
    sharding_config[f"transformer.h.{layer}.attn.c_proj"] = "row"
    sharding_config[f"transformer.h.{layer}.res_1"] = "replicated"
    sharding_config[f"transformer.h.{layer}.ln_2"] = "replicated"
    sharding_config[f"transformer.h.{layer}.mlp.c_fc"] = "column"
    sharding_config[f"transformer.h.{layer}.mlp.c_proj"] = "row"
    sharding_config[f"transformer.h.{layer}.res_2"] = "replicated"
    sharding_config[f"transformer.ln_f"] = "replicated"


with open (f"sharding_config_data_size_{value1}.dill", "rb") as f:
    data1 = dill.load(f)

# print combined table. col0: keys, col1: data1, col2: data2
print(tabulate([(k, sharding_config[k], data1[k]) for k in data1.keys()], headers=["Keys", f"Megatron-LM", f"Data Size {value1}"], tablefmt="grid"))