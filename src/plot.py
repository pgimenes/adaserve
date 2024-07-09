import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import re


def plot_optimizer_profiling(file="out.txt"):
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
    ax1.set_ylim([2300, 2600])
    ax1.set_xlim([0, 6000])
    ax1.grid(True)

    # Second subplot with bounds on the y-axis
    ax2.plot(times, bounds, marker="o", linestyle="-")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Dual Objective bounds")
    ax2.set_title("Dual Objective bounds over Time")
    ax2.set_xlim([0, 6000])
    ax2.grid(True)

    ax3.plot(times, gaps, marker="o", linestyle="-")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Percentage")
    ax3.set_title("Percentage gap between primal and Dual Objective over Time")
    ax3.set_xlim([0, 6000])
    ax3.grid(True)

    # Adjust layout
    plt.tight_layout()

    plt.savefig("experiments/bert_single_layer_bs_10k.png")


def plot_bs_seq_len(results):
    data = [[point[0][0], point[0][1], point[1]] for point in results]
    x, y, z = zip(*data)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Calculate throughput
    z = x * y / z

    # Remove all values where x or y is 1
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        if x[i] != 1 and y[i] != 1:
            nx.append(x[i])
            ny.append(y[i])
            nz.append(z[i])

    x = nx
    y = ny
    z = nz

    # Create grid data for interpolation
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method="cubic")

    # Create the figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the 3D scatter plot
    ax.scatter(x, y, z, c="b", marker="o")  # Blue color, circle marker

    # Plot the interpolated surface
    ax.plot_surface(xi, yi, zi, cmap="viridis", alpha=0.9)

    # Set axis labels
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Input sequence length")
    ax.set_zlabel("Single batch throughput (TPS)")

    # Add a title (optional)
    ax.set_title("3D Data Plot with Interpolated Surface")

    # Rotate the plot for better viewing (optional)
    ax.view_init(elev=15, azim=60)  # Adjust elevation and azimuth for desired view

    # Show the plot
    plt.savefig("experiments/sweep/sweep_batch_size_sequence_length.png")
    plt.show()
