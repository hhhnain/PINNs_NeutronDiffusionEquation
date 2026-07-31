import matplotlib.pyplot as plt
import numpy as np

# Load errors
data = np.loadtxt('all_errors.txt', skiprows=1)  # shape (27, 3)
rmse = data[:, 0]
mean_error = data[:, 1]
max_error = data[:, 2]

# Labels and color scales
titles = ["RMSE", "Mean Error (%)", "Max Error (%)"]
datasets = [rmse, mean_error, max_error]
cmaps = ["viridis", "plasma", "inferno"]

# Determine global color limits for consistency
vmins = [min(rmse), min(mean_error), min(max_error)]
vmaxs = [max(rmse), max(mean_error), max(max_error)]

# Create figure with 3 rows (metrics) × 9 columns (27 points)
fig, axes = plt.subplots(3, 9, figsize=(18, 6))

for row, (vals, title, cmap, vmin, vmax) in enumerate(zip(datasets, titles, cmaps, vmins, vmaxs)):
    for i, ax in enumerate(axes[row]):
        im = ax.imshow([[vals[i]]], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"#{i+1}", fontsize=8)
        ax.axis("off")

    # Add a single colorbar per row
    cbar = fig.colorbar(im, ax=axes[row].ravel().tolist(), orientation='horizontal',
                        fraction=0.04, pad=0.04)
    cbar.set_label(title)

plt.tight_layout()
plt.show()