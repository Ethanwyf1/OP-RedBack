# **On this page, you can find all notes and agendas from meetings with OPTIMA's clients.**

---


# **The Third Meeting Agendas**

**📅 Date:** April 08, 2025  
**📍 Participants:** Dr. Mario, Team Members  
**🗺️ Location:** Virtual & In-Person at The University of Melbourne -  Melbourne Connect

**💻 Zoom Link:** [Join Zoom Meeting](https://unimelb.zoom.us/j/81809831026?pwd=c222FVybp1Lbabu6bYXRtzeaaO5mdP.1)  

---

### **1️⃣ Project Progress Overview (10 min)**

> - Present the current status of ISA prototype development  
> - Highlight completed modules: Preprocessing, Workflow Decoupling, File Management, UX Design, Homepage  
> - Summarize implemented features and improvements since the last meeting

---

### **2️⃣ Open Q&A and Discussion (20 min)**

> - Address any questions or clarifications from Dr. Mario  
> - Discuss remaining challenges or unresolved issues  
> - Gather feedback on current implementation and suggestions for next steps  
> - Align on priorities for upcoming sprint

---
# **The Third Meeting Minutes**

**📅 Date:** April 09, 2025  
**📍 Participants:** Dr. Mario, Team members  

**Meeting Title:**  
Discussion and Clarification on ISA Prototype with Dr. Mario

---

## 🧩 Module 1: Preprocessing Stage

**🎯 Objective:**  
Load the uploaded dataset, clean missing data, and provide initial visualizations to prepare inputs for later stages.

**✅ Requirements & Decisions:**

- ✅ **Data Cleaning (Done)**  
  - Remove rows with excessive missing values  
  - Show clear labels like “Number of rows with missing values”  

- ✅ **Visualization (Done)**  
  - Generate boxplots and density plots  
  - Users can select multiple features to visualize via checkboxes or dropdowns (instead of text input)  

- ✅ **UI Improvements (Done)**  
  - Remove `feature_` and `algo_` prefixes from labels  
  - Add tooltips and status messages (e.g., upload success/failure, unexecuted prior stages)  

- ✅ **Export Options**  
  - Support downloading plots as `.png`  
  - Enable export of cleaned tables as `.csv`  

- ❗ **Note:**  
  - Normalization and feature selection are **not** handled here; they occur in Prelim stages  

---

## 🔄 Module 2: Stage Control and Workflow Decoupling

**🎯 Objective:**  
Enable modular execution of each ISA stage to support flexibility, transparency, and iterative development.

**✅ Requirements & Decisions:**

- ✅ **Independent Stage Execution**  
  - Each stage (Prelim, SIFTED, Pilot, Pythia, etc.) should run independently  
  - Provide a clean input/output interface between stages  

- ✅ **Caching Mechanism**  
  - Cache results after each stage  
  - Subsequent stages can reuse cached data  

- ✅ **Stage Navigation**  
  - Non-linear stage access is allowed (e.g., go directly to Trace)  
  - Add optional “Previous” / “Next” buttons  
  - Show warnings like “Please run Stage X first” when skipping dependencies  

- ✅ **Error & Status Feedback**  
  - Display loading spinners and status messages  
  - Provide error messages when processing fails or inputs are missing  

---

## 📁 Module 3: File Management and Workspace Handling

**🎯 Objective:**  
Standardize dataset uploads and outputs across all stages and allow session persistence.

**✅ Requirements & Decisions:**

- ✅ **Centralized Data Upload**  
  - Add a single upload entry point on the homepage or sidebar  
  - Uploaded datasets should be available to all stages  

- ✅ **Download Functionality**  
  - Allow downloading CSV outputs and plots for each stage  
  - Enable exporting consolidated data from all completed stages  

- ✅ **Workspace Management**  
  - Provide a way to reset dataset and cache  
  - Consider adding “Save Session” and “Load Session” for continuity  

---

## 🧭 Module 4: User Experience and Interaction Design

**🎯 Objective:**  
Ensure the tool is intuitive through better instructions, error handling, and stage explanations.

**✅ Requirements & Decisions:**

- ✅ **Navigation & Instructions**  
  - Each stage should have a brief explanation of its purpose and functionality  
  - Warn users if a required prior stage hasn’t been executed  

- ✅ **Error & Status Messaging**  
  - When no output is shown (e.g., due to unexecuted prior stages), display helpful messages  
  - Show success/failure notifications for uploads and processing  

- ✅ **Documentation**  
  - Add a help or “About” section with:  
    - Tool overview  
    - Explanation of each stage  
    - Suggested usage and common troubleshooting  

---

## 🏠 Module 5: Homepage Design

**🎯 Objective:**  
Provide an informative and functional landing page that guides users through the tool.

**✅ Requirements & Decisions:**

- ✅ **Feature Overview Section**  
  - Introduce the tool’s goals (e.g., visualizing algorithm performance on instance space)  
  - Describe each ISA stage in brief  

- ✅ **Workflow Visualization**  
  - Use flowcharts or step-by-step guides to illustrate how the tool works  
  - Include clickable links to each stage  

- ✅ **User Operation Guide**  
  - Guide users on how to upload data (supported format, structure, etc.)  
  - Clarify that skipping stages may result in missing output  

- ✅ **Data Upload Entry**  
  - Add a prominent upload button  
  - Show uploaded file name  
  - Automatically guide user to the first stage after upload  

- ✅ **Additional Suggestions**  
  - Add “FAQ” or “Contact Us” section  
  - Make homepage default landing screen  
  - Redirect users back to the homepage if no data is loaded
