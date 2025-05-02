# **The Fourth Meeting Agendas**

**📅 Date:** April 15, 2025  
**📍 Participants:** Dr. Mario, Team Members  
**🗺️ Location:** Virtual & In-Person at The University of Melbourne – Melbourne Connect  
**💻 Zoom Link:** [Join Zoom Meeting](https://unimelb.zoom.us/j/81809831026?pwd=c222FVybp1Lbabu6bYXRtzeaaO5mdP.1)

---

### **1️⃣ Key Discussion Points**

> 1. Upload & Preprocessing Workflow  
> 2. Caching System  
> 3. Data Flow Between Stages  

---

# **The Fourth Meeting Minutes**

**📅 Date:** 16/04/2024  
**📍 Participants:** Dr. Mario, Team Members  

---

📌 **Key Discussion Points**

1. **Upload & Preprocessing Workflow**

● Upload functionality is placed on the homepage to ensure the user provides a file before accessing the preprocessing stage.  

● The uploaded file is previewed with the first few columns and a confirmation message.  

● Preprocessing performs summarization, feature/algorithm selection, and includes:  
  ○ Boxplot visualizations (log scale and outlier toggle supported)  
  ○ Expandable details for features  
  ○ CSV export of preprocessed data  
  ○ Caching to preserve state across page refreshes  

---

2. **Caching System**

● Cache stores the uploaded metadata and preprocessing results in temporary `.cache` files.  

● Enables smooth transitions across sessions and supports data handoff to subsequent stages.  

● A delete button and optional timestamp will be added to inform users of the cache's state.  

---

3. **Data Flow Between Stages**

● Prelim stage will consume output from Preprocessing via the cache.  

● Current implementation avoids modifying the `pyInstanceSpace` functions directly.  

● Cache output is used to maintain consistency, especially after user-selected filtering.  

---

🧠 **Dr. Mario's Feedback**

✅ **Positive Observations:**

● Thoughtful implementation of preprocessing and caching logic.  

● Good use of interactive controls for visualizations and feature selection.  

---

⚠ **Recommendations:**

1. **Reuse Existing Package Functionality:**  
  ○ Before implementing custom solutions, examine if the desired feature (e.g., filtering) already exists in the `pyInstanceSpace` package.  
  ○ Noted that PreprocessingStage may support feature selection via `feature_names` and `algorithm_names` inputs.  

2. **Documentation Expectations:**  
  ○ Clearly document how each class is used, especially when interacting with `pyInstanceSpace`.  
  ○ Include what challenges were encountered and how they were solved.  
  ○ This will help him better understand your design choices and validate the tool.  

3. **Use Provided Test Cases:**  
  ○ The repo includes example input/output data for each stage.  
  ○ These can be used to test individual classes independently and support parallel development.  

---

🚧 **Next Steps**

🔄 **Development Plan**  
● Prelim Stage: Continue implementation using cached preprocessing output.  
● SIFTED Stage: To be developed using data from the original `pyInstanceSpace` repo.  

🧪 **Actions:**  
● Double-check if `pyInstanceSpace` supports feature selection internally and consider integrating it.  
● Continue using cache for consistency between stages unless native solutions offer significant benefits.  
● Schedule and conduct internal tests using example datasets.  
● Update documentation to reflect usage of each stage/class, especially regarding data handling and transitions.  

📅 **Timeline:**  
● Begin Prelim and SIFTED development in parallel.  
● Next meeting proposed on Wednesday next week (during Easter break) to confirm transition logic and move to the next 4 stages.  
