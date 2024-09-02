import subprocess

# Define the range for VLLM_AUTOSHARDING_DATA_SIZE
DATA_SIZES = [4, 8, 16, 32, 64, 128]
DATA_SIZES += list(range(256, 256 * 17, 256))

# Iterate through the range and execute the bash script with the specified VLLM_AUTOSHARDING_DATA_SIZE
for value in DATA_SIZES:
    print(f"Running value: {value}")

    # Execute the bash script with the environment variable
    with open (f"logs/autosharding_out_{value}.log", "w") as file, open(f"logs/autosharding_err_{value}.log", "w") as err:
        result = subprocess.run(f"VLLM_AUTOSHARDING_DATA_SIZE={value} ./experiments/benchmark_vllm.sh", shell=True, stdout=file, stderr=err)