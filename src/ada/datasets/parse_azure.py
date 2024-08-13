import pandas as pd

df = pd.read_csv("AzureLLMInferenceTrace_conv.csv")

# Convert TIMESTAMP column to datetime objects
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

# Subtract the first timestamp from all timestamps
df["TIMESTAMP"] = (df["TIMESTAMP"] - df["TIMESTAMP"].iloc[0]).dt.total_seconds()

df.to_csv("AzureLLMInferenceTrace_conv_parsed.csv")
