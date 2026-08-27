# Task 1: Depth vs Time Animation
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Load the data
data = pd.read_csv('depth_data.csv')

# Normalize column names: strip whitespace, so stray spaces don't break lookups
data.columns = data.columns.str.strip()

# The CSV has 'Point' (sample index) and 'Depth (m)', not 'time'/'depth'.
# Also, 'Depth (m)' contains at least one bad cell ("#VALUE!") which makes
# pandas read the whole column as text instead of numbers -> force it numeric,
# turning any unparseable entries (like "#VALUE!") into NaN.
data['Depth (m)'] = pd.to_numeric(data['Depth (m)'], errors='coerce')

data = data.dropna()  # drops rows with NaN (e.g. the #VALUE! row)

# extract point (used as our x-axis / "time") and depth columns
time = data['Point'].to_numpy()
depth = data['Depth (m)'].to_numpy().copy()

# outlier removal (local, not global) - compares each point to a rolling
# median of its neighbors so genuine dive peaks/troughs aren't mistaken for
# outliers just because they're far from the overall median.
depth_series = pd.Series(depth)
rolling_median = depth_series.rolling(window=9, center=True, min_periods=1).median()
outlier_mask = (depth_series - rolling_median).abs() > 100
depth_series[outlier_mask] = np.nan
depth = depth_series.to_numpy()

# Handle NaN values after outlier removal
noise_red_data = (
    pd.Series(depth).rolling(window=4, center=True)
    .mean()
    .bfill()
    .ffill()
    .to_numpy()
)

# Create the plot
win, grph = plt.subplots()
grph.grid(True)
grph.set_xlim(time.min(), time.max())
grph.set_ylim(noise_red_data.min() - 5, noise_red_data.max() + 5)
grph.set_xlabel("Point")
grph.set_ylabel("Depth (m)")
grph.set_title("Depth vs Point")

# Initialize the line object to be updated in the animation
line, = grph.plot([], [])

# Define the update function for the animation
def update(frame):
    line.set_data(time[:frame + 1], noise_red_data[:frame + 1])
    return line,

# Create the animation
animation = FuncAnimation(
    win,
    update,
    frames=len(time),
    interval=100,
    blit=True,
    repeat=False
)

# Show the plot
plt.show()
