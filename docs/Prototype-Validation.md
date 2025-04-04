## Validation Video

At Link: https://drive.google.com/file/d/1FeAbeLGcnJ3PejyYGmex046gNYno_tfa/view?usp=sharing

## Feedback From Industry Partner
###  General Feedback
- The 7-stage layout of the Streamlit interface was clear and well-received.
- Dr. Mario emphasized that each stage should support user interactivity, iterative runs, and clear visual feedback.

###  Feedback for each Stage
#### 1. Preprocessing Stage
Positive Feedback: Dataset summary was appreciated.

Improvements Suggested:

- Display key stats: number of instances, features, and algorithms.
- Add feature distribution visualizations (boxplots/bar charts).
- Allow toggling of preprocessing settings (e.g., normalization).
- Support iterative application with live result updates.

#### 2. Prelim Stage
Feedback Highlights:
- Combines normalization and binary label generation.
- Thresholds (e.g., performance cutoff) should be user-configurable.
- Visualize the distribution of assigned binary labels.
- Support repeat runs with modified settings.

#### 3. Sifted Stage

Important Clarification:
- Although technically optional, this step should be treated as mandatory.
- Provide an option to skip if curated features are already available.

Visualization & UI Suggestions:
- Use heatmaps or ranking plots to show selected features.
- Add a toggle to enable/disable feature selection.

#### 4. Pilot Stage

Core Function:

- Projects high-dimensional features into 2D for visualization.

Feedback & Suggestions:

- Display both Z (2D coordinates) and A (projection weights) matrices.
- Plot instances as points in 2D; ensure equal axis scaling.
- Support tooltips, color coding by performance/labels/clusters.
- Add form-based UI for parameter configuration.
- Enable reruns with modified settings. 

#### 5. Cloister Stage

Feedback:
- Should overlay the outer boundary of the instance cloud on the 2D plot.
- Can be executed independently.

#### 6. Trace Stage

Feedback:
- Overlay footprint polygons showing where each algorithm excels.
- Display footprint stats: area, density, purity.
- Use color-coded overlays and summary tables.

#### 7. Pythia Stage

Feedback:
- Show model performance metrics (accuracy, precision, recall).
- Optionally overlay predicted best algorithms on the 2D space.
- Use bar charts or tables for performance summaries.
