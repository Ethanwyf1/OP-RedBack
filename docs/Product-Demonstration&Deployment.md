# 🧠 Product Deployment & Usage Guide

## ✅ Live App Deployment

Our project has been successfully deployed using [Streamlit Community Cloud](https://streamlit.io/cloud).  
🔗 **Access the app here:**  
👉 [https://op-redback.streamlit.app](https://op-redback-tuahhkcddbtcktosbnztbg.streamlit.app)

---

## 🏁 Features Completed in Sprint 2

By the end of **Sprint 2**, we have completed and deployed the following:

- 🏠 **Homepage (Upload Interface)**:
  - Users can upload a `.csv` metadata file via a drag-and-drop or file selection interface.
  - The app checks for valid file type and format before processing.

- 🧹 **Preprocessing Stage**:
  - Handles normalization and initial cleaning.
  - Triggered after a successful metadata upload.
  - Feedback is shown in real time after preprocessing.

---

## 📂 How to Use the App

1. **Visit the App**  
   👉 [Launch App](https://op-redback-tuahhkcddbtcktosbnztbg.streamlit.app)

2. **Download the Sample Metadata File**  
   You **must** upload a valid metadata CSV file to proceed.  
   We provide a sample metadata file for testing:
   📄 [Download `metadata.csv`](https://github.com/user-attachments/files/20003470/metadata.1.csv)

3. **Upload the File in the App**  
   - <img width="906" alt="image" src="https://github.com/user-attachments/assets/cf43957a-f367-4ad8-85e0-00c15b693cf4" />
   
   - Use drag-and-drop or "Browse files" to upload the downloaded `metadata.csv` on the **Home page**
   - The app will begin preprocessing automatically

4. **Run Preprocessing**  
   - Use the **Feature Selection** and **Algorithm Selection** dropdowns to customize preprocessing.
   - Click **Run Preprocessing** to start data transformation and view real-time visual feedback.

5. **Download Processed Output**  
   - After preprocessing is complete, scroll down and click the **Download Cached Preprocessing Output (ZIP)** button to retrieve the processed data.

---

## 💡 Notes for Users

- File must be `.csv` format (UTF-8 encoded)
- Ensure required headers (e.g., `id`, `feature1`, `feature2`, …) are present
- Max size recommended: < 200MB

---

📅 *This wiki was last updated at the end of Sprint 2.*
- As the data transfer process has not been fully tested, we have decided to deploy the Prelim and Sifted stages in the next phase to ensure system stability. The current demo has covered the complete homepage and preprocessing stages and meets the requirements of the demo at this stage
