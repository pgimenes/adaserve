import re
import matplotlib.pyplot as plt

# Initialize lists to store the extracted values
times = []
bounds = []
solutions = []
gaps = []

# Define the regex pattern to match the lines
pattern = re.compile(
    r"^\s*[RL]?\s*\d+\s+\d+\s+\d+\s+\d+\.?\d*%\s+(\d+\.\d+|inf)\s+(\d+\.\d+|inf)\s+(\d+\.?\d*%)\s+\d+\s+\d+\s+\d+\s+\d+k?\s+(\d+\.\d+)s"
)

# Open and read the file line by line
with open("out.txt", "r") as file:
    for line in file:
        match = pattern.match(line)
        if match:
            # Extract the numbers using the capturing groups in the regex
            bound = float(match.group(1))
            solution = float(match.group(2))
            percentage_gap = float(
                match.group(3).strip("%")
            )  # Convert percentage to float
            time = float(match.group(4))

            # Append the extracted values to the respective lists
            bounds.append(bound)
            solutions.append(solution)
            gaps.append(percentage_gap)
            times.append(time)

# Plot the data
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# First subplot with solutions on the y-axis
ax1.plot(times, solutions, marker="o", linestyle="-")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Primal Solution")
ax1.set_title("Primal Objective Solution over Time")
ax1.set_ylim([74, 76])
ax1.grid(True)

# Second subplot with bounds on the y-axis
ax2.plot(times, bounds, marker="o", linestyle="-")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Dual Objective bounds")
ax2.set_title("Dual Objective bounds over Time")
ax2.grid(True)

ax3.plot(times, gaps, marker="o", linestyle="-")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Percentage")
ax3.set_title("Percentage gap between primal and Dual Objective over Time")
ax3.grid(True)

# Adjust layout
plt.tight_layout()

plt.savefig("output_figure.png")
