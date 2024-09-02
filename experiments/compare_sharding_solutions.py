import dill

value1 = 4
value2 = 4096

with open (f"sharding_config_data_size_{value1}.dill", "rb") as f:
    data1 = dill.load(f)

with open (f"sharding_config_data_size_{value2}.dill", "rb") as f:
    data2 = dill.load(f)

# use tabulate
from tabulate import tabulate

# print combined table. col0: keys, col1: data1, col2: data2
print(tabulate([(k, data1[k], data2[k]) for k in data1.keys()], headers=["Keys", f"Data Size {value1}", f"Data Size {value2}"], tablefmt="grid"))