# **On this page, you can find all notes and agendas from meetings with OPTIMA's clients.**

---

# **The Second Meeting Agendas**

**📅 Date:** April 1, 2025  
**📍 Participants:** Dr. Mario, Team Members  
**🗺️ Location:** Virtual Meeting Room & In-Person at The University of Melbourne, Melbourne Connect, 290-8-8108-Meeting Room (8)  
**💻 Zoom Link:** [Join Zoom Meeting](https://unimelb.zoom.us/j/81002139317?pwd=MsG3DQGvZdwjwkva0tFWRQs5Gd2c5f.1)

---

### **1️⃣ User Stories / Use Cases Presentation (10 min)**

> - Present the refined User Story Map covering the ISA pipeline stages:  
>   **Preprocessing → Prelim → Sifted → Pilot → Cloister → Pythia → Trace**  
> - Emphasize how each stage is designed to align with Stakeholders' needs  
> - Explain the corresponding use cases and expected user interactions

---

### **2️⃣ Prototype Demonstration (5 min)**

> - Show the updated Streamlit-based prototype  
> - Highlight key features such as data visualization, interactive configuration, and iterative execution  
> - Demonstrate how the prototype supports the outlined user stories and use cases

---

### **3️⃣ Discussion & Feedback Session (30 min)**

> - Open the floor for Industry Partners and Stakeholders to provide feedback  
> - Discuss potential adjustments to ensure alignment with Stakeholders’ needs  
> - Address any questions or concerns regarding the user stories and prototype functionality  
> - Identify action items and confirm next steps for further prototype refinement

---

### **Questions & Open Discussion Points**

> 1. Do the presented user stories and use cases fully meet the Stakeholders' needs?  
> 2. How effectively does the prototype demonstrate the envisioned functionality?  
> 3. What additional features or modifications are required?  
> 4. Are there any concerns regarding the iterative workflow and user interaction?  
> 5. What are the key areas for further enhancement based on today’s feedback?


---

# **The Second Meeting Minutes**

**📅 Date:** April 2, 2025 
**📍 Participants:** Dr. Mario, Team members  

**Meeting Title:**  
 Discussion and Clarification on ISA Prototype with Dr. Mario

---

1. **User Stories & Requirements**  
   ● The team presented a User Story Map outlining the ISA pipeline's seven stages.  
   
   ● Each stage (Preprocessing → Prelim → Sifted → Pilot → Cloister → Pythia → Trace)  
   corresponds to a user-driven analytical step, with visualizations between them to 
   support understanding of data transformation.
   
   ● The goal is to enable researchers to observe how the data and performance 
   evolve across stages and rerun specific stages if needed.
   
   ● **User needs:**  
     ○ Visualize raw and transformed data.  
     ○ Modify stage-specific options.  
     ○ Rerun stages iteratively.  
     ○ Navigate back and forth between stages.

2. **Prototype & Visualization (Detailed Per Stage)**  
   During the meeting, we demonstrated our Streamlit-based prototype, which structures the 
   user interface around the seven stages of the ISA pipeline. Dr. Mario provided targeted 
   feedback for each stage regarding what should be visualized and how users should interact with the system: 

   1. **Preprocessing Stage**  
      ● **Function:** Cleans and standardizes the raw data, filling missing values and 
      normalizing features.  
      
      ● **Visualization Suggestions:**  
        ○ Display key dataset statistics (e.g., number of instances, features, and 
        algorithms).  
        ○ Use boxplots or bar charts to show feature distributions.  
        ○ Optionally display before/after transformation comparisons for selected 
        features.  
      
      ● **Interactivity:**  
        ○ Allow users to toggle settings (e.g., normalization) and rerun this stage.  
        ○ Users can repeatedly apply different preprocessing settings and see the 
        impact live.

   2. **Prelim Stage**  
      ● **Function:** Converts algorithm performance into binary labels (good/bad) and 
      performs outlier bounding.  
      
      ● **Feedback:**  
        ○ This stage combines normalization and binary label generation.  
        ○ Users should be able to adjust thresholds (e.g., performance cutoff values).  
      
      ● **Suggestions:**  
        ○ Show the distribution of binary labels assigned to the instances.  
        ○ Make thresholds configurable through the UI and support iterative application.

   3. **Sifted Stage**  
      ● **Function:** Performs feature selection based on correlation.  
      
      ● **Algorithm Details (from Dr. Mario):**  
        ○ Calculate correlations between each feature and the performance.  
        ○ Then, group correlated features together and select only one from each 
        group.  
      
      ● **Feedback:**  
        ○ This is a recommended but optional(but we need to assume it is mandatory) 
        step — users may skip this if they already have curated features.  
      
      ● **Suggestions:**  
        ○ Use heatmaps or feature ranking plots to visualize selected features.  
        ○ Provide the option to toggle feature selection on/off.

   4. **Pilot Stage – Dimensionality Reduction**  

      **Function**  
      The Pilot stage performs dimensionality reduction, transforming high-dimensional instance 
      feature vectors into a 2D space using a linear projection approach (similar to PCA, but 
      typically optimized via BFGS). This transformation allows instances to be visualized as 
      individual points in a 2D instance space, enabling comparison, clustering, and further 
      analysis in subsequent stages. 
![Picture 1](https://github.com/user-attachments/assets/1aa9889c-33a5-4801-bd72-5c14d9a6931b)

      
      **Outputs**  
      1. **Z Matrix – Projected 2D Coordinates**  
         ● **Shape:** n_instances × 2  
         ● **Meaning:** For each instance, this matrix contains its coordinates in the 2D projection 
         space.  
         ● **Use:** This is the primary output for visualization — each row is a point plotted on the 
         2D scatter plot.  
         ● **Remarks:**  
           ○ Z is analogous to the PCA result X' = XW (where W is the projection matrix).  
           ○ This is what is shown visually to the user as the 2D "instance space".  
      
      2. **A Matrix – Projection Weights Matrix**  
         ● **Shape:** 2 × n_features  
         ● **Meaning:** This matrix defines the linear transformation from the original 
         high-dimensional feature space to the 2D space.  
         ● **Mathematical form:**  
         Z=X⋅ATorzi=A⋅xi(for each instance)\text{Z} = \text{X} \cdot A^T \quad \text{or} \quad 
         \text{z}_i = A \cdot \text{x}_i \quad \text{(for each instance)}Z=X⋅ATorzi =A⋅xi (for each 
         instance)  
         ● **Use:**  
           ○ Can be interpreted as a 2D “projection direction” for each original feature.  
           ○ Helpful for visual diagnostics or interpreting the contribution of each feature.  
         ● **Visualization (optional):**  
           ○ Display A as a heatmap or arrow overlay (if doing biplot-style projections).  
           ○ Display a numeric matrix table in a collapsible panel for advanced users.
      
      **Visualization of Outputs**  
      - **2D Scatter Plot of Z**  
        ● Each point = one instance.  
        ● Points are plotted on a 2D plane using the rows of Z.  
      
      - **Optional enhancements:**  
        ○ Tooltip showing instance name, original feature values, or algorithm 
        performance.  
        ○ Color coding by:  
          ■ Algorithm performance  
          ■ Cluster/grouping  
          ■ Binary label (from Prelim)  
      
      - **Use as a Base Plot**  
        ● This instance space serves as the foundation for later overlays in:  
          ○ Trace → adds footprint polygons for algorithm performance.  
          ○ Cloister → adds outer boundary of instance distribution.  
          ○ Pythia → can overlay the predicted best algorithm per point.  
      
      - **Axis Scaling**  
        ● Dr. Mario emphasized the importance of equal aspect ratio:  
          "These plots should look square, not squashed. Axes should have equal scale."  
      
      **Interactivity & Iteration**  
      - **User-defined parameters:**  
        ○ e.g., num_tries, analytic (whether to use analytic gradients), initial_points  
      - These are passed in as a dictionary (pilot_options) to the Pilot class.  
      - **Dr. Mario confirmed:**  
        "Every stage is a class with its own set of options. You can run and rerun the stage 
        by modifying these parameters."  
      
      - **Suggested UI Elements:**  
        ● A form to edit Pilot options (text inputs, dropdowns).  
        ● An "Apply" button to trigger the stage computation.  
        ● The 2D scatter plot panel showing the new Z matrix.  
        ● Optional: collapsible section for viewing the A matrix.

   5. **Cloister Stage**  
      ● **Function:** Defines the outer boundary (bounding box or polygon) of the instance 
      cloud in the projected 2D space.  
      ● **Feedback:**  
        ○ This stage is independent and can be executed separately.  
        ○ Helps identify the region covered by the dataset.  
      ● **Visualization:**  
        ○ Overlay the computed boundary on the Pilot plot to show coverage.

   6. **Trace Stage**  
      ● **Function:** Identifies and visualizes the regions (footprints) where each algorithm 
      performs well.  
      ● **Outputs:**  
        ○ Footprint polygons, along with statistics like area, density, and purity.  
      ● **Feedback:**  
        ○ Must overlay footprint polygons on the Pilot 2D space.  
        ○ Each footprint represents a region where a specific algorithm is highly 
        effective.  
      ● **Visualization:**  
        ○ Color-coded regions with boundaries, along with a summary table of stats.

   7. **Pythia Stage**  
      ● **Function:** Trains predictive models (e.g., SVMs) to recommend the best algorithm 
      for unseen instances.  
      ● **Outputs:**  
        ○ Model performance metrics (accuracy, precision, recall).  
        ○ Predicted labels or probabilities per instance.  
      ● **Feedback:**  
        ○ Display prediction performance using bar charts or tables.  
        ○ Optionally overlay predictions on the 2D space to illustrate where each 
        algorithm is likely to succeed.

3. **Data Transformation & ISA Pipeline Clarification (Per-Stage Execution Logic)**  
   Dr. Mario offered an in-depth clarification of how the pipeline should function and how the 
   user interface should support iterative, stage-by-stage execution.

   **General Principles:**  
   ● Each ISA stage is implemented as an independent Python class with its configurable options (usually defined in a JSON structure).  
   ● The goal is to enable users to adjust parameters for any stage and re-run that stage without needing to rerun the entire pipeline.  
   ● The tool should support a modular and interactive workflow, where users:  
     ○ Modify stage-specific options.  
     ○ Execute only that stage via an "Apply" button.  
     ○ Observe visualized results.  
     ○ Decide to proceed or revise inputs again.

   **Pipeline Flexibility:**  
   ● Cloister can run independently.  
   ● Pythia and Trace can both derive from Pilot, and are not dependent on each other.  
   ● Trace and Pythia can be re-run multiple times using different parameters.  
   ● Returning to a previous stage (e.g., Prelim or Sifted) to reapply processing is expected and supported.

   **UI Design Implications:**  
   ● Each stage page should have:  
     ○ A configuration section with editable inputs (text boxes, toggles).  
     ○ A visualization pane showing the result of that stage.  
     ○ An Apply button to rerun that stage with the current config.  
   ● For some stages, allow options like:  
     ○ Changing number of trials (e.g., in Pilot).  
     ○ Toggling normalization or bounding flags (e.g., in Prelim).  
     ○ Displaying raw vs processed data for comparison (e.g., in Preprocessing).  
   ● Inputs must be passed into each class as Python dictionaries (i.e., options), which should be collected through the UI.  
   ● The final output of each stage should update the pipeline's state and support branching or returning as needed.

   **Key Insight from Dr. Mario:**  

   > "The most important feature is not just visualization—it's the ability to iterate over individual stages, adjust parameters, and see the effect without redoing the whole analysis."

4. **Functionality Design**  
   ● **Key required functionalities:**  
     ○ Modifiable configuration options per stage, editable via GUI (text inputs, checkboxes).  
     ○ Apply button to rerun a specific stage with new parameters.  
     ○ Visualization panel per stage:  
       ■ For Preprocessing: show boxplots, data summaries.  
       ■ For Prelim: show binary performance stats, transformation results.  
       ■ For Pilot: visualize 2D instance space (e.g., scatter plot with coordinates).  
       ■ For Trace: footprint boundary plots + statistics (area, density, purity).  
       ■ For Pythia: model accuracy charts and prediction outputs.  
     ○ Iterative workflow navigation:  
       ■ Users can return to previous stages, modify settings, and reprocess data without restarting the entire pipeline.

5. **Technical Stack Discussion**  
   ● **Preferred Framework:** Streamlit  
     ○ Easy to integrate backend and frontend.  
     ○ Rapid prototyping with minimal learning curve.  
     ○ Supports required functionality: file uploads, user inputs, data visualizations.  
   ● **Alternatives considered:**  
     ○ Dash: more customizable but heavier to implement.  
     ○ React + FastAPI: powerful but time-consuming for a student project.  
   ● **Dr. Mario approved the choice:**  
     ○ Functionality > Aesthetics.  
     ○ Prioritize iteration, parameter tweaking, and visualization clarity.  
     ○ Acceptable to use Streamlit if it leads to a working prototype with necessary features.

6. **UI Design Example**

<img width="521" alt="23343345" src="https://github.com/user-attachments/assets/335f07b2-ae2c-44f6-9df2-a80e1242e090" />



7. **Workflow**

<img width="1440" alt="Screenshot 2025-04-04 at 7 23 54 AM" src="https://github.com/user-attachments/assets/ae967254-4e8d-443f-9643-c30d2f76e2bf" />



   > **Hint：** Trace does not depend on Pythia's output, but on Pilot.  
   > It is correct to say that both Trace and Pythia come from Pilot and can be run independently of each other.  
   > However, Trace can optionally use Pythia's predictions for footprint evaluation (this is an enhancement, not a requirement).


