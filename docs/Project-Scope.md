
## Project Scope

### Table of Contents
- [Overview](#overview)
- [Stakeholder-Analysis](Project-Scope#Stakeholder-Benefits-Analysis)
- [Project Motivations](#project-motivations)
- [Scope Boundaries](#scope-boundaries)
  - [In Scope](#in-scope)
  - [Out of Scope](#out-of-scope)
- [Success Criteria](#success-criteria)
- [Constraints](#constraints)

### Overview
The MATILDA Python-Native Dashboard project encompasses the development of a complete web interface for the Instance Space Analysis workflow, replacing the existing MATLAB-based solution. This document defines clear boundaries for what this project will and will not address.

### Project Motivations
The current MATLAB-based solution presents several challenges for users, including:
- Difficulty tracking intermediate results through the analysis pipeline

By transitioning to a Python-native web interface, this project aims to:
- Democratize access to powerful algorithm selection tools
- Improve research efficiency through better visualization and workflow tracking
- Enhance collaboration capabilities through modern web technologies


## Stakeholder Benefits Analysis

### Stakeholder Analysis Capabilities

This Python dashboard will enable various stakeholders to perform several types of analysis that deliver specific benefits:

#### Academic Researchers
- **Algorithm Performance Comparison**: Researchers can visualize how different algorithms perform across various problem instances, identifying strengths and weaknesses of each approach.
- **Instance Space Mapping**: The ability to project high-dimensional instance features onto a 2D space, allowing researchers to understand the structure of problem domains.
- **Feature Selection and Impact Analysis**: Researchers can select different feature sets and observe their impact on instance space structure, enabling more informed methodological decisions.
- **Algorithm Footprint Visualization**: The dashboard will show regions where specific algorithms excel, helping researchers understand algorithm applicability domains.

#### Data Scientists in Industry
- **Automated Algorithm Selection**: Practitioners can use the dashboard to identify which algorithms are most suitable for their specific problem instances.
- **Performance Prediction**: The system will enable prediction of algorithm performance on new, unseen instances based on their features.
- **Parameter Sensitivity Analysis**: Users can explore how parameter adjustments affect algorithm performance across different regions of the instance space.
- **Instance Generation Guidance**: The dashboard will help identify areas of the instance space that lack coverage, guiding the generation of new benchmark instances.

#### OPTIMA Platform Administrators
- **Usage Pattern Analysis**: Administrators can track which features and algorithms are most frequently explored by users.
- **System Performance Monitoring**: The dashboard will provide insights into computational resource usage and processing times.
- **Content Management**: Administrators can more easily update algorithm libraries and instance features without requiring MATLAB expertise.

### Stakeholder-Feature Benefit Mapping

| Stakeholder | Feature | Expected Benefit | Measurement Method |
|-------------|---------|------------------|-------------------|
| Academic Researchers | Interactive Instance Space Visualization | More intuitive understanding of problem structure leading to better research hypotheses | Number of citations to papers using the platform |
| Academic Researchers | Custom Feature Selection | Ability to test novel features and their impact on instance characterization | Number of new features proposed and shared in the community |
| Data Scientists | Algorithm Recommendation Engine | Reduced time to select appropriate algorithms for specific problems | Time saved in algorithm selection process (survey) |
| Data Scientists | Batch Processing Interface | Ability to analyze large datasets without manual intervention | Volume of instances processed per analysis session |
| OPTIMA Administrators | Python-based Dashboard | Easier maintenance and extension compared to MATLAB-based tools | Reduction in maintenance time and increased feature development rate |
| OPTIMA Administrators | User Activity Analytics | Better understanding of user needs to guide future development | User retention and engagement metrics |



### Scope Boundaries

#### In Scope
* Development of a web-based dashboard interface compatible with modern browsers
* Integration with the existing instancespace Python package
* Implementation of visualizations for all key stages of the analysis workflow
* User authentication and project management capabilities
* Data import/export functionality supporting common file formats
* Interactive result exploration with filtering and comparison tools
* Documentation and user guides for both end-users and future developers

#### Out of Scope
* Modifications to core algorithms within the instancespace package
* Changes to the underlying MATILDA platform infrastructure
* Integration with external analysis tools beyond the current workflow

### Success Criteria
* Complete processing of standard workflows previously handled by MATLAB
* User feedback indicating improved workflow visibility
* Comparable or improved performance metrics
* Enhanced accessibility of intermediate processing results

### Constraints

| **Constraint** | **Justification** |
|----------------|-------------------|
| **Team & Timeline** | Our team consists of 5 students working on the project during a 12-week university semester. Alongside this project, team members have other academic and personal commitments. This limits available time for development, testing, and documentation, requiring careful planning and prioritization. |
| **Client Collaboration** | We are working closely with a client (Dr. Mario), and some development tasks depend on receiving timely feedback and clarifications. Delays in client responses can slow down progress or lead to assumptions being made, affecting accuracy or design decisions. |
| **Technical Environment** | The dashboard must integrate with the `pyInstanceSpace` Python package without changing its internal logic. We are required to use Python and Streamlit to build the dashboard, which are beginner-friendly but come with some limitations in layout customization and advanced control. |
| **Workflow Dependencies** | The system is built as a pipeline with multiple stages (Preprocessing → Prelim → Sifted → etc.). Each stage depends on the output of the one before it. We must ensure data is passed properly between stages, and caching is implemented so earlier steps don’t have to be repeated every time. |
| **Resource Constraints** | Full testing of all features might not be possible within the project timeline. Some advanced features like exporting visualizations or optimizing interactivity may be skipped if time is limited. Task assignments are based on team members' comfort levels with frontend, backend, or documentation. |
