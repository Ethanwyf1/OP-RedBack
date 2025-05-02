# **The Fifth Meeting Agendas**

📅 **Date:** April 23, 2025  
📍 **Participants:** Zihang Xu, Dr. Mario, Yifei Wang, Abdulrahman Alaql  
🗺️ **Location:** Virtual & In-Person at The University of Melbourne – Melbourne Connect  
💻 **Zoom Link:** [Join Zoom Meeting](#)

---

## **1️⃣ Key Discussion Points**

- Prelim Stage Visualization
- Sifted Stage Visualization
- Areas for Improvement


---

## 1️⃣ Prelim Stage Discussion

### Progress Update (Zihang Xu)
- Added an optional checkbox allowing users to specify whether higher values are better for performance.
- Integrated message box tooltips to guide users' selections.
- Implemented initial visualizations for:
  - Algorithm-feature relationships
  - Feature correlations
  - Performance distribution among algorithms

### Feedback from Dr. Mario
- **Clarify Purpose of Visualizations**:
  - Every plot should have an accompanying explanation describing what it represents and why it is useful.
  - Descriptions should be concise, similar to annotations typically added in Python notebooks.
- **Algorithm Naming**:
  - Replace generic labels like "Algorithm 1", "Algorithm 2" with actual algorithm names for better user understanding.
- **Download Figures (Optional Enhancement)**:
  - Consider allowing users to download individual figures as images if time permits.
- **Feature Importance Explanation**:
  - Clearly explain how feature importance is determined, especially if based on metrics like maximum correlation values.

---

## 2️⃣ Sifted Stage Discussion

### Progress Update (Zihang Xu)
- Completed running the Sifted stage, observing that the process is very slow due to large parameter sets and computations.
- Generated outputs including:
  - Selected features Importance
  - Correlation Heatmap（feature vs algorithm)
  - Silhouette scores
  - Clustering results
- Used preprocessing outputs for feature labels, since Prelim does not generate its own.

### Feedback from Dr. Mario
- **Performance Expectations**:
  - It is normal for the Sifted stage to take a long time because of the extensive parameter tuning and feature evaluations.
- **Selective Visualization**:
  - Not all outputs need to be visualized.
  - Focus on the most critical outputs:
    - Selected features
    - Raw scores
    - Silhouette scores
    - Clustering results
- **Highlight Key Outputs**:
  - Raw and silhouette scores are particularly important, as they directly influence subsequent analysis and decision-making.
- **Reuse of Feature Labels**:
  - It is acceptable and correct to reuse feature labels from the preprocessing stage for downstream stages like Prelim and Sifted.
- **Options Reuse**:
  - Stick to one instance of `selvars_options`.
  - Every stage will require the `selvars_options` variable, so it is acceptable to keep only one instance.

---

## 📌 Additional Key Discussion Points

- Selected Feature Importance Histogram
- Present Correlation between features and algorithms in heatmap
- Silhouette Scores vs. Number of Clusters
- PCA projection in 2D

---

## Areas for Improvement

- It is encouraged to add explanation text along the visualizations for users to have a better understanding of the image.
- Instead of using `feature1`, `feature2` vs `algo1`, `algo2`, it is better to list the feature and algorithm names.
- `clust` output from the Sifted Stage is worth visualizing.

---

## ⚠️ Recommendations

- Stick to one instance of `selvars_options`: Every stage will require the `selvars_options` variable, so it is acceptable to keep only one instance.

---

## 🚧 Next Steps

### 🔄 Development Plan
- Remove extra `selvars_options`.
- SIFTED Stage: Add explanation texts for each graph and clust visualization.

---

## 3️⃣ Other Notes

- Participants confirmed the plan for next week’s meeting at the same time.
- Yifei Wang is going to make a plan for the Pilot stage, with updates expected in the next meeting.
