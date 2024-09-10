import pandas as pd

# Conv
# =====================

df = pd.read_csv("src/ada/datasets/AzureLLMInferenceTrace_conv.csv")

# Convert TIMESTAMP column to datetime objects
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

# Subtract the first timestamp from all timestamps
df["TIMESTAMP"] = (df["TIMESTAMP"] - df["TIMESTAMP"].iloc[0]).dt.total_seconds()

df.to_csv("src/ada/datasets/AzureLLMInferenceTrace_conv_parsed.csv")

# Code
# =====================

df = pd.read_csv("src/ada/datasets/AzureLLMInferenceTrace_code.csv")

# Convert TIMESTAMP column to datetime objects
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

# Subtract the first timestamp from all timestamps
df["TIMESTAMP"] = (df["TIMESTAMP"] - df["TIMESTAMP"].iloc[0]).dt.total_seconds()

df.to_csv("src/ada/datasets/AzureLLMInferenceTrace_code_parsed.csv")
