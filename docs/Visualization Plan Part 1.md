
# 📊 Visualization Plan – Part 1

## 🏠 Module 0: Homepage (New)

### 🎯 Objective
Introduce the ISA platform to users, explain its functionality, and guide them on how to begin.

### ✅ Key Content to Include
- A brief introduction to Instance Space Analysis and its benefits.
- Overview of the 7-stage pipeline:  
  `Preprocessing → Prelim → Sifted → Pilot → Cloister → Pythia → Trace`
- Step-by-step instructions on uploading metadata (e.g., CSV format, structure).
- Traditional algorithm benchmarking problems and how ISA improves on them.

### ✨ UI/UX Enhancements to Implement
- Add **"Getting Started"** or **"How it Works"** sections with diagrams.
- Include a sample dataset download and upload panel.
- Use collapsible panels or cards to organize explanations for each pipeline stage.

---

## 🧩 Module 2: Prelim Stage

### 🎯 Objective
Transform algorithm performance into binary labels (`good`/`bad`) and normalize feature values to bound outliers and prepare for classification.

### ✅ Confirmed Functions
- Generate binary labels based on a user-defined threshold.
- Normalize feature values.
- Reduce the effect of outliers.

### 📌 User Feedback & Suggestions
- Threshold values should be configurable via the UI.
- Normalization and label assignment should be iteratively controllable.
- Show a distribution chart of binary labels to help users understand class balance.

### ✨ UI/UX Enhancements to Implement
- Add sliders or numeric inputs for threshold configuration.
- Display bar chart or histogram of binary label distribution.
- Provide status messages if this stage hasn’t been executed yet.

---

## 🧩 Module 3: Sifted Stage

### 🎯 Objective
Select the most informative features by analyzing correlation with performance and remove redundant features via clustering.

### ✅ Confirmed Functions
- Compute correlation between features and performance.
- Cluster correlated features and select one representative.

### 📌 User Feedback & Suggestions
- Make this stage mandatory (though technically optional).
- Allow toggle for feature selection (for advanced users).
- Visualize selected vs. dropped features using heatmaps or ranking plots.

### ✨ UI/UX Enhancements to Implement
- Add toggle to enable/disable this stage.
- Use correlation heatmaps, feature importance charts.
- Display selected vs. excluded features.

---

## 📋 Task Breakdown – Prelim & Sifted Stage Visualization

### 🟩 Data Processing Tasks (Prefix: `D`)

| ID  | Task                            | Description                                                             | Dependency |
|-----|---------------------------------|-------------------------------------------------------------------------|------------|
| D1  | Define Preprocessing Output     | Specify data fields needed for Prelim and Sifted                        | None       |
| D2  | Implement Cache Output          | Save outputs using `save_cleaned_metadata()`                            | D1         |
| D3  | Load Cached Data                | Load metadata for visualization using `load_cleaned_metadata()`         | D2         |
| D4  | Create MockDataManager          | Generate mock inputs (X_mock, Y_mock, labels_mock)                      | None       |

### 🟦 Prelim Stage UI Tasks (Prefix: `P`)

| ID  | Task                            | Description                                                             | Dependency |
|-----|---------------------------------|-------------------------------------------------------------------------|------------|
| P1  | Threshold Config UI            | Sliders/input for `performance_cutoff` and `beta_threshold`             | None       |
| P2  | Label Distribution Visualization | Show binary label counts and `num_good_algos`                           | P1, D4     |
| P3  | Good Algorithm Count Chart      | Show `num_good_algos` per instance                                      | P1, D4     |
| P4  | Execution Button                | "Run Prelim" updates visuals and logs                                   | P1, D3     |
| P5  | Summary Display                 | Summary text/logs of outlier removal and normalization                  | P4         |

### 🟨 Sifted Stage UI Tasks (Prefix: `S`)

| ID  | Task                            | Description                                                             | Dependency |
|-----|---------------------------------|-------------------------------------------------------------------------|------------|
| S1  | Parameter Selection UI          | Dropdown to select algorithm performance                                | None       |
| S2  | Correlation Heatmap             | Visualize feature-performance correlation                               | S1, D4     |
| S3  | Kept vs Dropped Features        | Show features retained and discarded                                    | S2         |
| S4  | Clustering Toggle Option        | Enable clustering if applicable                                         | S3         |
| S5  | Execution Button                | "Run Sifted" button with logs                                           | S1, D3     |

### 🧩 Common Support Tasks (Prefix: `A`)

| ID  | Task                            | Description                                                             | Dependency |
|-----|---------------------------------|-------------------------------------------------------------------------|------------|
| A1  | Execution Status Check          | Show warning if prior stage not executed                                | D3         |
| A2  | Mock Data Toggle                | Enable dev mode to switch between real and mock data                    | D4         |

---

## 🔄 Two Approaches for Stage-to-Stage Data Handling

### ✅ Option A: Use Original Functions

Use existing instance space functions (e.g., `from_csv_file()`).

```python
metadata = from_csv_file("data/metadata.csv")
```

### ✅ Option B: Use Mock Data

Simulate previous outputs using mock files and `st.session_state`.

```python
with open("mock_outputs/prelim_mock_output.pkl", "rb") as f:
    prelim_output = pickle.load(f)
```

### 📊 Comparison Table

| Criteria                        | Option A                          | Option B                         |
|---------------------------------|-----------------------------------|----------------------------------|
| ✅ Getting Started Speed         | Fast                              | Slightly slower                  |
| 🔄 Sync with Previous Stage      | Poor                              | Controlled and accurate          |
| 🔧 Dev Flexibility               | High                              | High                             |
| 🧪 Consistency with Final Output | ❌ Risk of mismatch                | ✅ Mimics final behavior          |
| 🔁 Refactor Cost                 | Medium                            | Low                              |
| 👥 Collaboration Support         | Moderate (custom setups)          | Excellent (shared mock data)     |

---

## 👥 Development vs. School Workflow Balance

### 🧑‍🤝‍🧑 Team: 5 Members
### 🎯 Goal: Build visualization for ISA stages while fulfilling university deliverables

### ✅ Current Strategy
- **Short-term target**: PRELIM and SIFTED stage visualization
- **Yifei's focus**: Planning & scaffolding later stages (PILOT → TRACE)
- All members involved in both dev and academic tasks

---

## 🔀 Role Allocation Plans

### 🅰️ Allocation 1 – Balanced Participation

**Assignment**
- Team 1: PRELIM  
- Team 2: SIFTED  
- Everyone contributes to the academic work

**Pros**
- ✅ Hands-on experience for all  
- ✅ Better knowledge sharing  
- ✅ Strong alignment between product and report

**Cons**
- ⚠️ Context switching may slow down work  
- ⚠️ Coordination needed to avoid conflicts

---

### 🅱️ Allocation 2 – Specialized Roles

**Assignment**
- 2 members: Development only (1 per stage)  
- 2 members: Academic writing only

**Pros**
- ✅ Higher technical depth  
- ✅ More efficient execution  
- ✅ Clear responsibility boundaries

**Cons**
- ⚠️ Dev-team may become siloed  
- ⚠️ Report team disconnected from technical flow

---

## 📅 Deadline

- **Submission Deadline**: **April 23**
- **Team Allocation**:
  - `Anujan` & `Muath`: Prelim  
  - `Zihang` & `Abdul`: Sifted

