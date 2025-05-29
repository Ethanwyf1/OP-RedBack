from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point     # ← add this line
from instancespace.data.options import ParallelOptions, TraceOptions
from instancespace.stages.trace import TraceInputs, TraceStage, TraceOutputs
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from matplotlib.colors import to_rgba


algo_labels_path = "external/pyInstanceSpace/tests/test_data/trace_csvs/algolabels.txt"

# Use Path.open() to open the file
with open(algo_labels_path, 'r') as f:

    algo_labels = f.read().split(',')

# Reading instance space from Z.csv
z = np.genfromtxt(
    "external/pyInstanceSpace/tests/test_data/trace_csvs/Z.csv",
    delimiter=",",
    dtype=np.double,
)

# Reading binary performance indicators from y_bin.csv
y_bin = np.genfromtxt(
    "external/pyInstanceSpace/tests/test_data/trace_csvs/yhat.csv",
    delimiter=",",
    dtype=np.int_,
).astype(np.bool_)

# Reading binary performance indicators from y_bin2.csv
y_bin2 = np.genfromtxt(
    "external/pyInstanceSpace/tests/test_data/trace_csvs/yhat2.csv",
    delimiter=",",
    dtype=np.int_,
).astype(np.bool_)

# Reading performance metrics from p.csv
p1 = np.genfromtxt(
    "external/pyInstanceSpace/tests/test_data/trace_csvs/selection0.csv",
    delimiter=",",
    dtype=np.double,
)
p1 = p1 - 1  # Adjusting indices to be zero-based

# Reading performance metrics from p2.csv
p2 = np.genfromtxt(
    "external/pyInstanceSpace/tests/test_data/trace_csvs/dataP.csv",
    delimiter=",",
    dtype=np.double,
)
p2 = p2 - 1  # Adjusting indices to be zero-based

# Reading beta thresholds from beta.csv
beta = np.genfromtxt(
    "external/pyInstanceSpace/tests/test_data/trace_csvs/beta.csv",
    delimiter=",",
    dtype=np.int_,
).astype(np.bool_)

# Setting TRACE options with a purity value of 0.55 and enabling sim values
trace_options = TraceOptions(True, 0.55)

parallel_options = ParallelOptions(False, 3)

# Initialising and running the TRACE analysis
trace_inputs: TraceInputs = TraceInputs(
    z,
    p1.astype(np.double),
    p2.astype(np.double),
    beta,
    algo_labels,
    y_bin,
    y_bin2,
    trace_options,
    parallel_options,
)

trace_output: TraceOutputs = TraceStage._run(trace_inputs)
good_fps = trace_output.good

def draw_region(ax, poly, facecolor, edgecolor, label):
    """Fill & outline a single Polygon or MultiPolygon on ax."""
    if poly is None:
        return
    if isinstance(poly, Polygon):
        pieces = [poly]
    else:
        pieces = list(poly.geoms)
    for i, p in enumerate(pieces):
        x, y = p.exterior.xy
        ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor, alpha=0.4,
                label=label if i==0 else None)

n_algos = len(algo_labels)
cols    = 3
rows    = (n_algos + cols - 1) // cols
space_poly  = trace_output.space.polygon

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
axes = axes.flatten()

for i, name in enumerate(algo_labels):
    ax       = axes[i]
    good_poly = good_fps[i].polygon

    # compute point‐masks
    # inside overall space:
    if space_poly:
        space_mask = np.array([space_poly.contains(Point(x,y)) for x,y in z])
    else:
        space_mask = np.ones(len(z), dtype=bool)
    # inside this algo's good footprint:
    if good_poly:
        good_mask = np.array([good_poly.contains(Point(x,y)) for x,y in z])
    else:
        good_mask = np.zeros(len(z), dtype=bool)
    # bad = in space but not in good
    bad_mask = space_mask & ~good_mask

    # draw the regions
    bad_region = space_poly.difference(good_poly) if space_poly and good_poly else None
    draw_region(ax, bad_region,  facecolor="lightgray", edgecolor="gray",  label="bad region")
    draw_region(ax, good_poly,    facecolor="red",       edgecolor="darkred", label="good region")

    # scatter the points
    ax.scatter(z[bad_mask,0], z[bad_mask,1], s=10, c="black", label="bad points")
    ax.scatter(z[good_mask,0], z[good_mask,1], s=10, c="red",   label="good points")

    ax.set_title(name)
    ax.set_xlabel("Z₁")
    ax.set_ylabel("Z₂")
    ax.legend(markerscale=2)

plt.tight_layout()
plt.show()


# best_fps = trace_output.best
# fig, ax = plt.subplots(figsize=(8, 6))


# for i, name in enumerate(algo_labels):

#     best_poly = best_fps[i].polygon

#     if space_poly:
#         space_mask = np.array([space_poly.contains(Point(x,y)) for x,y in z])
#     else:
#         space_mask = np.ones(len(z), dtype=bool)

#     # inside this algo's good footprint:
#     if best_poly:
#         good_mask = np.array([best_poly.contains(Point(x,y)) for x,y in z])
#     else:
#         good_mask = np.zeros(len(z), dtype=bool)

#     draw_region(ax, best_poly, facecolor="red", edgecolor="darkred", label="good region")

#     ax.scatter(z[good_mask,0], z[good_mask,1], s=10, c='r',   label="good points")

#     ax.set_title(name)
#     ax.set_xlabel("Z₁")
#     ax.set_ylabel("Z₂")
#     ax.legend(markerscale=2)

# plt.tight_layout()
# plt.show()
