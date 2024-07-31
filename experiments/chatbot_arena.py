from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import defaultdict

dataset = "lmsys/lmsys-chat-1m"

ds = load_dataset(dataset)
ds = ds["train"]


# Step 1: Extract timestamps and sequence lengths
timestamp_sequence_length = defaultdict(int)
timestamp_batch_size = defaultdict(int)

ROUNDING = 1
MAX_CONVERSATION_LENGTH = 1

for entry in ds:
    timestamp = round(entry["tstamp"] / ROUNDING) * ROUNDING
    conversation_a_length = sum(
        len(turn["content"])
        for turn in entry["conversation_a"][:MAX_CONVERSATION_LENGTH]
    )
    conversation_b_length = sum(
        len(turn["content"])
        for turn in entry["conversation_b"][:MAX_CONVERSATION_LENGTH]
    )
    total_length = conversation_a_length + conversation_b_length
    timestamp_sequence_length[timestamp] = max(
        total_length, timestamp_sequence_length[timestamp]
    )
    timestamp_batch_size[timestamp] += 1

# Step 2: Prepare data for plotting
timestamps = list(timestamp_sequence_length.keys())
sequence_lengths = list(timestamp_sequence_length.values())
batch_sizes = list(timestamp_batch_size.values())

print(max(timestamps) - min(timestamps))

# Step 3: Plot the data (as a figure with 2 subfigures)
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# Plot 1: Sequence Lengths
axs[0].plot(timestamps, sequence_lengths)
axs[0].set_title("lmsys/chatbot_arena_conversations")
axs[0].set_xlabel("Timestamp")
axs[0].set_ylabel("Sequence Length")

# Plot 2: Batch Sizes
axs[1].plot(timestamps, batch_sizes)
axs[1].set_xlabel("Timestamp")
axs[1].set_ylabel("Batch size")

plt.tight_layout()
plt.show()
