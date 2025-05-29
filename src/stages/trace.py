import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point     # ← add this line
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from utils.cache_utils import cache_exists, load_from_cache
from instancespace.data.options import ParallelOptions, TraceOptions
from instancespace.stages.trace import TraceInputs, TraceStage
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from matplotlib.colors import to_rgba
from utils.cache_utils import cache_exists, delete_cache, load_from_cache, save_to_cache


def show():
    st.header("🧬 TRACE Stage")

#     # --- prerequisite caches ---
#     for dep, name in [
#         ("pilot_output.pkl", "Pilot"),
#         ("sifted_output.pkl", "SIFTED"),
#         ("prelim_output.pkl", "PRELIM"),
#     ]:
#         if not cache_exists(dep):
#             st.error(f"🚫 {name} output not found. Please run the {name} stage first.")
#             return

    

    # # 1) Let the user pick which y_bin to use
    # st.subheader("Select Resource")
    # choice = st.radio(
    #     "Select performance mask:",
    #     ("Real y_bin (from Prelim)", "Pythia y_bin (from Pythia)")
    # )

#     preprocessing_out = load_from_cache("preprocessing_output.pkl")
#     prelim_out = load_from_cache("prelim_output.pkl")
#     sifted_out = load_from_cache("sifted_output.pkl")
#     pilot_out = load_from_cache("pilot_output.pkl")

#     use_sim = False

#     z = pilot_out.z
#     selection0 = None 
#     p = None
#     beta = prelim_out.beta
#     algo_labels = preprocessing_out.algo_labels
#     y_hat = None
#     y_bin = None
#     trace_options = None
#     parallel_options = ParallelOptions(False, 3)
    
#     print(beta)

#     # 2) Based on choice, check cache
#     if choice == "Real y_bin (from Prelim)":
        
#         # load your real y_bin here, e.g.:
#         prelim_out = load_from_cache("prelim_output.pkl")
#         y_bin = prelim_out.y_bin
#         p = prelim_out.p

#     else:  # Pythia
#         if not cache_exists("pythia_output.pkl"):
#             st.error("🚫 Pythia output not found. Please run the PYTHIA stage first.")
#             return
#         # load your pythia y_bin here, e.g.:
#         pythia_out = load_from_cache("pythia_output.pkl")
#         y_hat = pythia_out.y_hat
#         selection0 = pythia_out.selection0
#         use_sim = True
    

#     st.subheader("Select Purity")
#     purity = st.slider(
#         label="Polygon purity threshold",
#         min_value=0.0,
#         max_value=1.0,
#         value=TraceOptions.default().purity,
#         step=0.01,
#         help=(
#             "Purity is the minimum fraction of ‘good’ points that a small region "
#             "must contain to be kept in the final footprint polygon."
#         )
#     )

#     trace_options = TraceOptions(use_sim=use_sim, purity=purity)

#     # run trace
#     trace_inputs: TraceInputs = TraceInputs(
#         z,
#         selection0,
#         p,
#         beta,
#         algo_labels,
#         y_hat,
#         y_bin,
#         trace_options,
#         parallel_options,
#     )


#     if st.button("🚀 Run TRACE", key="run_trace_btn"):
#         trace_out = TraceStage._run(trace_inputs)
#         st.session_state["trace_output"] = trace_out
#         st.success("TRACE completed!")
#         st.write(trace_out)
#         st.write(trace_out.good[0])


#     if "trace_out" in st.session_state:
#         trace_out = st.session_state["trace_output"]
        
        

# def plot_polygon(ax, poly, edgecolor="black", facecolor="none", lw=2, label=None):
#     """Plot a Shapely Polygon or MultiPolygon on ax."""
#     if poly is None:
#         return
#     if isinstance(poly, Polygon):
#         polys = [poly]
#     else:
#         polys = list(poly.geoms)
#     for p in polys:
#         x, y = p.exterior.xy
#         ax.plot(x, y, color=edgecolor, lw=lw, label=label)
#         label = None

    # sel_algo = st.selectbox("Algorithm", algo_labels, key="single_algo")
    # idx = algo_labels.index(sel_algo)

    # 1) Let the user pick which y_bin to use
    st.subheader("Select Resource")
    choice = st.radio(
        "Select performance mask:",
        ("Real y_bin (from Prelim)", "Pythia y_bin (from Pythia)")
    )

    st.subheader("Select Purity")
    purity = st.slider(
        label="Polygon purity threshold",
        min_value=0.0,
        max_value=1.0,
        value=TraceOptions.default().purity,
        step=0.01,
        help=(
            "Purity is the minimum fraction of ‘good’ points that a small region "
            "must contain to be kept in the final footprint polygon."
        )
    )
    
    use_sim = False

    if choice != "Real y_bin (from Prelim)":
        use_sim = True
    

    algo_labels_path = "external/pyInstanceSpace/tests/test_data/trace_csvs/algolabels.txt"
    algo_labels = None
    
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
    trace_options = TraceOptions(use_sim, purity)
    parallel_options = ParallelOptions(False, 3)

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

    if st.button("🚀 Run TRACE"):

        st.info("Running TRACE...")
        trace_output: TraceOutputs = TraceStage._run(trace_inputs)
        save_to_cache(trace_output, "trace_output.pkl")
        st.session_state["trace_output"] = trace_output
        st.session_state["ran_trace"] = True
        st.toast("TRACE completed successfully!", icon="✅")

    if st.session_state.get("ran_trace", False) or cache_exists("trace_output.pkl"):
        trace_output = st.session_state.get("trace_output") or load_from_cache("trace_output.pkl")

        z = trace_inputs.z
        best_fps = trace_output.best
        good_fps = trace_output.good
        hard = trace_output.hard

        # good/bad footprints visualization
        st.subheader("Good Footprints")
        sel_algo = st.selectbox("Algorithm", algo_labels, key="select_good_algo")
        idx = algo_labels.index(sel_algo)
        good_poly = good_fps[idx].polygon

        fig1, ax1 = plt.subplots(figsize=(10,6))
        plot_good_fps(fig1, ax1, good_poly, trace_output.space.polygon, z, sel_algo)

        # hard footprint visualization
        st.subheader("Hard Region")
        hard_poly = hard.polygon
        fig2, ax2 = plt.subplots(figsize=(10,6))
        plot_hard_region(fig2, ax2, hard_poly, trace_output.space.polygon, z)

        
        st.subheader("Best Footprints")

        fig3, ax3 = plt.subplots(figsize=(10,6))
        plot_best_fps(fig3, ax3, best_fps, z, algo_labels)
        




def plot_good_fps(fig, ax, good_poly, space_poly, z, sel_algo):
    
    # space masks
    if space_poly:
        space_mask = np.array([space_poly.contains(Point(x, y)) for x,y in z])
    else:
        space_mask = np.ones(len(z), dtype=bool)

    if good_poly:
        good_mask = np.array([good_poly.contains(Point(x, y)) for x,y in z])
    else:
        good_mask = np.zeros(len(z), dtype=bool)

    bad_mask = space_mask & ~good_mask
    # bad region
    bad_region = space_poly.difference(good_poly) if space_poly and good_poly else None
    draw_region(ax, bad_region, facecolor="lightgray", edgecolor="gray",  label="bad region")
    # good region
    draw_region(ax, good_poly, facecolor="red", edgecolor="darkred", label="good region")
    # points
    ax.scatter(z[bad_mask,0],  z[bad_mask,1],  s=10, c="black", label="bad points")
    ax.scatter(z[good_mask,0], z[good_mask,1], s=10, c="red",   label="good points")

    ax.set_title(f"{sel_algo} — Good Footprint")
    ax.set_xlabel("Z₁")
    ax.set_ylabel("Z₂")
    ax.legend(markerscale=2)

    st.pyplot(fig)

def plot_hard_region(fig, ax, hard_poly, space_poly, z):
    
    if space_poly:
        space_mask = np.array([space_poly.contains(Point(x, y)) for x,y in z])
    else:
        space_mask = np.ones(len(z), dtype=bool)

    if hard_poly:
        hard_mask = np.array([hard_poly.contains(Point(x, y)) for x,y in z])
    else:
        hard_mask = np.zeros(len(z), dtype=bool)
    
    other_mask = space_mask & ~hard_mask
    # draw_region(ax, hard_poly, facecolor="red", edgecolor="darkred", label="hard region")

    ax.scatter(z[hard_mask,0], z[hard_mask,1], s=10, c="red", label="hard points")
    ax.scatter(z[other_mask,0], z[other_mask,1], s=10, c="black", label="other points")


    ax.set_title(f"Hard Footprint")
    ax.set_xlabel("Z₁")
    ax.set_ylabel("Z₂")
    ax.legend(markerscale=2)

    st.pyplot(fig)

def plot_best_fps(fig, ax, best_fps, z, algo_labels):
    
    for i, algo in enumerate(algo_labels):
        print(f"{i}: {algo}")
        best_poly = best_fps[i].polygon
        
        if not best_poly:
            continue

        color = f"C{i}"
        best_poly = best_fps[i].polygon

        if best_poly:
            best_mask = np.array([best_poly.contains(Point(x, y)) for x,y in z])
        else:
            best_mask = np.zeros(len(z), dtype=bool)
        
        # print("BEST_MASK")
        # print(best_mask)
        print(len(z[best_mask,0]))
        print(z[best_mask,0])
        if len(z[best_mask,0]) == 0:
            continue

        # draw_region(ax, best_poly, facecolor=color, edgecolor="darkred", label=f"{algo}_region")
        ax.scatter(z[best_mask,0], z[best_mask,1], s=10, c=color, label=algo)

    ax.set_title(f"Best Footprint")
    ax.set_xlabel("Z₁")
    ax.set_ylabel("Z₂")
    ax.legend(markerscale=2)
    st.pyplot(fig, use_container_width=True)
    

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


    # # ==============================

    # n_algos = len(algo_labels)
    # cols    = 3
    # rows    = (n_algos + cols - 1) // cols
    # space_poly  = trace_output.space.polygon

    # fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    # axes = axes.flatten()

    # for i, name in enumerate(algo_labels):
    #     ax       = axes[i]
    #     good_poly = good_fps[i].polygon

    #     # compute point‐masks
    #     # inside overall space:
    #     if space_poly:
    #         space_mask = np.array([space_poly.contains(Point(x,y)) for x,y in z])
    #     else:
    #         space_mask = np.ones(len(z), dtype=bool)
    #     # inside this algo's good footprint:
    #     if good_poly:
    #         good_mask = np.array([good_poly.contains(Point(x,y)) for x,y in z])
    #     else:
    #         good_mask = np.zeros(len(z), dtype=bool)
    #     # bad = in space but not in good
    #     bad_mask = space_mask & ~good_mask

    #     # draw the regions
    #     bad_region = space_poly.difference(good_poly) if space_poly and good_poly else None
    #     draw_region(ax, bad_region,  facecolor="lightgray", edgecolor="gray",  label="bad region")
    #     draw_region(ax, good_poly,    facecolor="red",       edgecolor="darkred", label="good region")

    #     # scatter the points
    #     ax.scatter(z[bad_mask,0], z[bad_mask,1], s=10, c="black", label="bad points")
    #     ax.scatter(z[good_mask,0], z[good_mask,1], s=10, c="red",   label="good points")

    #     ax.set_title(name)
    #     ax.set_xlabel("Z₁")
    #     ax.set_ylabel("Z₂")
    #     ax.legend(markerscale=2)

    # plt.tight_layout()
    # st.pyplot(fig)