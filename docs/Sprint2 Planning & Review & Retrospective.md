---

## 📅 **Sprint 2 Plan (Weeks 5-8)**

---

### 🟩 **Week 5: UI Mockups & Interaction Design**

**Objective:**  
Translate prioritized user stories into low- to high-fidelity UI mockups to align team understanding and guide implementation.

**Key Activities:**  
- Design wireframes for key visualization screens:
  - PV-1: Preprocessing → Prelim transition view
  - PV-2: Prelim → Sifted feature selection view
  - PV-3: Sifted → 2D projection transition (Pilot)
  - AA-1: Algorithm performance comparison (Pythia)
  - DE-1: Web-based dashboard layout
- Present mockups for team review and feedback
- Begin storyboarding user flow interactions

**Output:**  
- Figma or Miro prototypes covering core interactions
- Mockups included in Wiki documentation

---

### 🟦 **Week 6: Technology Stack Confirmation & Environment Setup**

**Objective:**  
Finalize technical tools and prepare the development environment for rapid iteration.

**Key Activities:**  
- Evaluate and confirm tech stack (e.g., Streamlit vs Dash)
- Set up:
  - Local development environment (Python, Git, Virtualenv)
  - Basic repo structure
  - CI/CD (optional)
- Create initial Streamlit boilerplate with placeholder pages

**Output:**  
- Confirmed tech stack with justification
- Shared development setup instructions (in Wiki)
- Initial “Hello World” app running locally

---

### 🟨 **Week 7: Learning & Experimenting with Stack**

**Objective:**  
Ensure team members are comfortable with the selected stack before implementation begins.

**Key Activities:**  
- Complete short learning tasks:
  - Streamlit layout, component usage, and callbacks
  - Plotly/Altair integration with dummy data
  - Multi-page app setup
- Each member builds a small test visualization independently
- Share & discuss lessons learned at week-end

**Output:**  
- Mini prototypes with working charts/interactions
- Better understanding of reusable components

---

### 🟧 **Week 8: Foundational Implementation Begins**

**Objective:**  
Start building working prototypes of core user stories (Release 1 targets).

**Stories Targeted for Development:**
- PV-1: Preprocessing → Prelim visualization  
- PV-2: Feature selection impact visualization  
- AA-1: Displaying performance metrics from Pythia  
- DE-1: Deployable web interface skeleton  
- INT-2: Hover interactions for instance identification  

**Key Activities:**
- Implement UI structure and layout
- Connect real data outputs (e.g., projection matrices, metadata)
- Integrate first visual elements (scatter plots, bar charts)
- Set up interactions (hover, dropdown filters)

**Output:**
- Initial working version of the dashboard
- Preview ready for user feedback and iteration

---

> ✅ **Sprint 2 Deliverable Goal:**  
Functional visualization skeleton covering the core ISA pipeline stages with mock or initial data.

## Sprint2 Review

---

### 1. Attendees
- **Product Owner: Muath**  
- **Scrum Master: Anujan**  
- **Development Team: Yifei, Zihang**  
- **Stakeholders / Guests: Dr.Andrés**  

---

### 2. Completed Work
- Construct the layout of the web-baed dashboard
- Preprocessing Stage With:
    - feature and algorithm selection feature
    - data summary table and feature distribution plot
    - Download Processed Data Option

- Prelim Stage With:
    - Binary Performance Summary Bar Chart
    - Best Algorithm Distribution
    - Raw vs. Processed Performance Density Plots
    - Feature Transformation Histograms
    - Feature Importance Ranking
    - Beta Selection Visualization
    - Download Processed Data Option

- Sifted Stage With :
    - Feature Importance Histgram
    - Correlation Heatmap（feature vs algorithm)
    - Silhouette curve
    - PCA 2D projection


---

### 3. Work Not Done

* **AA-1: Displaying Performance Metrics from Pythia**  
  Deferred pending further client input on the project stages; we prioritized completing the earlier stages first.


---

### 4. Feedback & Discussion
- Provide context for every plot. Include a brief description of what each figure shows and explain why it’s important.
- Use real algorithm names. Swap out placeholders like “Algorithm 1” and “Algorithm 2” for their actual names to make the results more transparent.
- Enable figure downloads if possible. If time allows, add an option for users to download individual plots as image files.
- Streamline SiftedStage outputs. Focus on the key results:
    - Selected features
    - Raw scores
    - Silhouette scores
    - Clustering results
- It is acceptable and correct to reuse feature labels from the preprocessing stage for downstream stages like Prelim and Sifted.

---

### 5. Action Items
---

| Action Item | Owner |
|-------------|-------|
| Add descriptive captions for every plot explaining what it shows and why it matters | Muath & Anujan & Zihang| 
| Replace placeholder names (“Algorithm 1”, “Algorithm 2”) with their actual algorithm names throughout the dashboard | Muath & Anujan & Zihang & Yifei|
| Implement a “Download as Image” button or link on each figure for user export | Muath & Anujan & Zihang & Yifei|
| Refine SiftedStage outputs to display only: <br>• Selected features<br>• Raw scores<br>• Silhouette scores<br>• Final clustering visualizations | Zihang |
| Ensure feature labels from PreprocessingStage are consistently reused in Prelim and Sifted stages | Zihang |



## Sprint2 Retrospective

---

### 1. What Went Well
* Goal Alignment: Completed most tasks in line with the provisional goals set during Sprint 2 planning.
* Continuous Progress: Team members worked proactively over the break to keep momentum.
* Client Validation: Successfully validated the product increment with the client, gathering positive feedback.
* Collaboration & Support: Strong team collaboration; members were supportive and communicative throughout the sprint.

---

### 2. What Didn’t Go Well

* Low Attendance at Validation: Only a few team members could attend the client validation session due to illness or lecture conflicts.
* Overextension Over Break: Working during the break led to fatigue for some team members.
* Planning Estimates: Some user stories were underestimated, impacting throughput.


---

### 3. Action Items

| Action Item | Owner |
|-------------|-------|
| Reschedule a follow-up client validation session at a time when everyone can attend | Muath |
| Review and adjust estimation process | Anujan |

---
