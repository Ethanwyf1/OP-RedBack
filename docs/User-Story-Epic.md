# 🚀 Project User Story

---

## Table of Contents
- [User Stories by Epic](#user-stories-by-epic)
  - [Process Visualization](#-process-visualization)
  - [Algorithm Analysis](#-algorithm-analysis)
  - [Footprint Analysis](#-footprint-analysis)
  - [Instance Analysis](#-instance-analysis)
  - [Usability](#-usability)
  - [Security](#-security)
  - [Performance](#-performance)
  - [Enhanced Filtering](#-enhanced-filtering)
  - [Deployment & Data Export](#-deployment--data-export)
  - [Documentation](#-documentation)
  - [Advanced Analysis](#-advanced-analysis)
  - [Interactivity](#-interactivity)
- [User Story Dependency Mapping](#-user-story-dependency-mapping)
- [Sprint 2 Plan (Weeks 5-8)](#-sprint-2-plan-weeks-5-8)
- [Technology Stack](#-technology-stack)
- [Next Steps & Actions](#-next-steps--actions)

---

## 📖 User Stories by Epic

# User Stories with Acceptance Criteria

## 🔄 Process Visualization

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| PV-1 | Dr. Li Wen | Visualize the transition from 'Preprocessing' to 'Prelim' stages | I can see and understand how the data is transformed during preprocessing before performance evaluation | Medium | Must Have | Helps researchers understand preprocessing behavior, enabling enhancement of preprocessing techniques | 1. The visualization must clearly display data before and after preprocessing with visual indicators highlighting transformations.<br>2. Key preprocessing metrics must be shown alongside the visualization.<br>3. Users must be able to select specific data points to see detailed transformation information.<br>4. The visualization must load within 3 seconds with standard datasets. |
| PV-2 | Dr. Li Wen | Visualize the transition from 'Prelim' to 'Sifted' stages | I can understand how binary performance measures inform feature selection | Medium | Must Have | Provides insight into relevant features for algorithm analysis | 1. The visualization must display which features were selected and which were discarded with performance measures that influenced feature selection clearly presented.<br>2. Users must be able to toggle between seeing all features and only selected features.<br>3. The visualization must include tooltips explaining the selection criteria for each feature and provide an overview panel showing summary statistics of the feature selection process. |
| PV-3 | Dr. Li Wen | Visualize the transition from 'Sifted' to 'Pilot' stages | I can see how feature selection impacts the projection to 2D space | Medium/Large | Must Have | Critical for relating problem characteristics to algorithm performance | 1. The visualization must show both pre-projection feature space and post-projection 2D space with visual links connecting related data points between the two spaces.<br>2. Key metrics about projection quality must be displayed including stress and variance explained.<br>3. Users must be able to adjust projection parameters and see real-time updates.<br>4. Color coding must highlight how different feature combinations affect projection results. |
| PV-4 | Sara Mansour | Visualize the PILOT projection matrix output | I can understand how instances are distributed in 2D space | Large | Must Have | Fundamental to subsequent analysis and visualization | 1. The 2D projection must clearly display all instances with appropriate scaling and axes labeled with meaningful information about what they represent.<br>2. Clusters of similar instances must be visually identifiable.<br>3. Users must be able to zoom, pan, and select regions of interest.<br>4. The projection must include reference points or boundaries for context and render correctly across different screen sizes. |

## 🟩 Algorithm Analysis

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| AA-1 | Alex Rivera | Visualize algorithm performance from PYTHIA stage | I can compare performance metrics across all 10 algorithms | Medium | Must Have | Enables direct comparison of algorithm effectiveness | 1. Performance metrics for all 10 algorithms must be displayed in a single comparative view, allowing users to sort and filter algorithms by different performance metrics.<br>2. The visualization must highlight the best-performing algorithm for each metric and indicate statistical significance of performance differences where appropriate.<br>3. Users must be able to toggle between tabular and graphical representations of performance data. |
| AA-2 | Alex Rivera | Visualize cross-validation results from PYTHIA | I can understand prediction accuracy (89.6%) and precision (88.7%) | Medium | Should Have | Helps assess reliability of algorithm predictions | 1. Cross-validation metrics must be clearly displayed for each algorithm with confidence intervals or error bars shown for all metrics.<br>2. Users must be able to drill down to see per-fold performance.<br>3. The visualization must highlight potential overfitting or high variance issues and provide comparison between validation and test performance. |

## 🟧 Footprint Analysis

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| FA-1 | Dr. Li Wen | Visualize algorithm footprints from TRACE stage | I can identify regions where specific algorithms excel | Large | Must Have | Core output showing algorithm strengths and weaknesses | 1. Footprints must be displayed as clearly defined regions in the 2D instance space with color coding differentiating between algorithms and performance levels.<br>2. Users must be able to toggle individual algorithm footprints on/off.<br>3. The visualization must include a legend explaining footprint interpretation with footprint boundaries rendered with appropriate smoothing and clarity.<br>4. The visualization must update within 2 seconds when changing displayed algorithms. |
| FA-2 | Dr. Li Wen | Visualize footprint metrics (Area_Good, Density_Best) | I can quantitatively compare algorithm coverage and effectiveness | Medium | Must Have | Provides objective measures of performance regions | 1. Metrics must be displayed alongside footprint visualizations with comparative bar or radar charts showing metrics across all algorithms.<br>2. Users must be able to set thresholds for what constitutes "good" coverage.<br>3. The visualization must highlight algorithms with complementary metrics and clearly display changes in metrics when adjusting parameters. |
| FA-3 | Alex Rivera | Compare multiple algorithm footprints simultaneously | I can identify complementary algorithms and performance gaps | Medium | Should Have | Enables selection of well-covered algorithm portfolios | 1. Users must be able to select 2+ algorithms for side-by-side or overlay comparison with overlapping and unique coverage areas visually distinguishable.<br>2. The visualization must highlight uncovered regions in the instance space and suggest complementary algorithm pairs based on coverage analysis.<br>3. Users must be able to save comparison views for future reference. |

## 🟪 Instance Analysis

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| IA-1 | Alex Rivera | Filter and select specific instances in visualizations | I can focus analysis on particular problem subsets | Medium | Should Have | Supports detailed investigation of algorithm behavior | 1. Users must be able to select instances by dragging, clicking, or using filter criteria with selected instances remaining highlighted across different visualizations.<br>2. Selection statistics must be displayed including count and average performance.<br>3. Users must be able to save and name selections for later analysis.<br>4. Filtering options must include both feature values and algorithm performance metrics. |

## 🟨 Usability

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| UB-1 | Sara Mansour | Adjust key parameters (e.g., beta_threshold) | I can see how parameter changes affect analysis results | Medium | Should Have | Allows exploratory analysis and sensitivity testing | 1. Parameter controls must be intuitive and accessible in the UI with appropriate min/max limits and step sizes.<br>2. Changes to parameters must trigger appropriate visualization updates.<br>3. The system must provide suggested parameter values for common scenarios.<br>4. Current parameter values must always be visible and resettable to defaults. |

## 🟫 Security

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| SEC-1 | Dr. Li Wen | Log in securely to the dashboard | Experiments and data is protected | Medium | Must Have | Only allows the authorized users to access the sensitive research data | 1. Login must require strong password credentials with failed login attempts limited and logged.<br>2. Sessions must timeout after 30 minutes of inactivity.<br>3. Two-factor authentication must be available for sensitive data.<br>4. Password reset functionality must be secure and user-friendly. |
| SEC-2 | Dr. Li Wen | Implement secure API endpoints | Data exchange between components is protected | Medium | Must Have | Prevents unauthorized data access and manipulation | 1. All API endpoints must require authentication tokens with API requests encrypted using HTTPS.<br>2. Rate limiting must be implemented to prevent abuse.<br>3. API access logs must be maintained for audit purposes.<br>4. Input validation must prevent injection attacks. |

## 🟪 Performance

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| PERF-1 | Alex Rivera | Access cached experiment results | I can view frequently used data quickly without recomputation | Medium | Should Have | Improves user experience and reduces computational overhead | 1. Frequently accessed data must load in under 1 second.<br>2. The system must intelligently cache based on usage patterns.<br>3. Users must be able to manually trigger cache refresh when needed.<br>4. Cache status indicators must show when data was last updated.<br>5. Memory usage must be optimized to prevent performance degradation.<br>6. When multiple users 200 and 500 access the dashboard at the same time, it need to be loaded within an exact time 2-5 seconds for most of users.<br>7. When many users try to use the system simultaneously and it becomes overloaded, it should show clear error messages rather than crashing.<br>8. The system should consistently meet performance in all supported browsers (Chrome, Firefox, Safari, Edge), even when many users access it simultaneously. |

## 🟩 Enhanced Filtering

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| EF-1 | Dr. Li Wen | Define custom metrics for filtering | I can analyze results according to my specific research needs | Medium/Large | Should Have | Enables more flexible and personalized data analysis | 1. Users must be able to create and name custom metrics using a formula builder that can combine existing metrics and mathematical operations.<br>2. Filtering interfaces must incorporate custom metrics alongside standard ones.<br>3. Custom metrics must persist between sessions.<br>4. Users must be able to share custom metrics with other researchers.<br>5. The system must validate custom metrics to prevent errors. |

## 🟥 Deployment & Data Export

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| DE-1 | Sara Mansour | Access the dashboard through a web interface | I can use it without installing specialized software | Large | Must Have | Accessible to a wider audience | 1. The dashboard must function correctly in major browsers including Chrome, Firefox, Safari, and Edge.<br>2. The interface must be responsive and adapt to different screen sizes.<br>3. Initial page load must complete in under 5 seconds on standard connections.<br>4. Offline mode must provide limited functionality when connection is lost.<br>5. Browser requirements and limitations must be clearly documented. |
| DE-2 | Alex Rivera | Export visualizations and analysis results | I can include them in publications and presentations | Small | Should Have | Facilitates knowledge sharing and dissemination | 1. Users must be able to export visualizations in common formats including PNG, SVG, and PDF.<br>2. Exported images must have appropriate resolution for publication.<br>3. Data tables must be exportable to CSV or Excel formats.<br>4. Exported content must include relevant captions and metadata.<br>5. Batch export of multiple visualizations must be supported. |

## 🟫 Documentation

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| DT-1 | Sara Mansour | Access explanations of ISA methodology alongside visualizations | I can properly interpret the results | Medium | Should Have | Enhances educational value and ensures correct interpretation | 1. Contextual help must be available for each visualization type.<br>2. Documentation must explain key ISA concepts and terminology and include examples of correct interpretation.<br>3. Documentation must be searchable by keyword.<br>4. Users must be able to provide feedback on documentation clarity. |
| DT-2 | Sara Mansour | Follow a step-by-step tutorial | I can quickly learn to use the tools | Medium | Should Have | Reduces learning curve and improves adoption | 1. Tutorial must cover all basic functionality in a logical sequence with each step including clear instructions and screenshots.<br>2. Tutorial progress must be saved between sessions.<br>3. Users must be able to skip or revisit tutorial sections as needed.<br>4. The tutorial must include interactive elements for hands-on learning. |

## 🟦 Advanced Analysis

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| AA-ADV-1 | Dr. Li Wen | Process multiple datasets in batch mode | I can efficiently analyze multiple domains | Large | Could Have | Increases productivity for multi-dataset analysis | 1. Users must be able to queue multiple datasets for processing with batch jobs running in the background without blocking the UI.<br>2. Progress indicators must show status of each dataset in the batch.<br>3. Email notifications must be sent when batch processing completes.<br>4. Results from batch processing must be organized for easy comparison.<br>5. Error handling must ensure partial failures don't stop the entire batch. |

## 🟨 Interactivity

| Epic | As a | I Want To | So That | Size Estimation | MoSCoW Priority | Justification | Acceptance Criteria |
|------|------|-----------|---------|-----------------|-----------------|---------------|---------------------|
| INT-1 | Alex Rivera | Toggle between algorithm visualizations | I can focus on specific algorithms without visual clutter | Small | Should Have | Improves usability when analyzing many algorithms | 1. The interface must include clear toggle controls for each algorithm that update visualizations within 1 second.<br>2. Current toggle state must always be visible.<br>3. Users must be able to save favorite toggle configurations.<br>4. Keyboard shortcuts must be available for quick toggling. |
| INT-2 | Sara Mansour | Hover over points in instance space for details | I can identify specific instances and characteristics | Small | Must Have | Essential for interactive exploration | 1. Hover tooltips must appear within 200ms and display instance ID and key metrics.<br>2. Tooltips must be positioned to avoid obscuring relevant data.<br>3. Users must be able to pin tooltips to keep them visible while exploring nearby points.<br>4. Information density in tooltips must be configurable. |

-----
## 🔗 **User Story Dependency Mapping**

| User Story | Depends On |
|------------|------------|
| PV-1       | None       |
| PV-2       | PV-1       |
| PV-3       | PV-2       |
| PV-4       | PV-3       |
| AA-1       | PV-4       |
| AA-2       | AA-1       |
| FA-1       | AA-1       |
| FA-2       | FA-1       |
| FA-3       | FA-1       |
| IA-1       | PV-4       |
| UB-1       | AA-1, FA-1 |
| INT-1      | AA-1       |
| INT-2      | PV-4       |
| DE-1       | None       |
| DE-2       | PV-4, AA-1, FA-1 |
| DT-1       | DE-1       |
| DT-2       | DE-1       |
| SEC-1      | None       |
| SEC-2      | SEC-1      |
| PERF-1     | PV-4, AA-1, FA-1 |
| EF-1       | IA-1, UB-1 |
| AA-ADV-1   | All core functionalities completed |


## 🔍 **Technology Stack**

- **Framework:** Streamlit

---

## 📆 **Next Steps & Actions**

- Finalize and confirm UI and technology stack (Weeks 5-6)
- Prepare environment and tutorials (Week 7)
- Begin development (Week 8)
