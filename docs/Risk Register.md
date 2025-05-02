# 🛡️ Risk Register

This document outlines the identified project risks, categorized under **General Development Risks**, **Ethical Considerations**, and **Cyber Security Considerations**. Each risk is evaluated by Probability (P), Impact (I), and their resulting Risk Score (P × I).

---

## 📑 Table of Contents

- [⚙️ General Development Risks](#-general-development-risks)
- [🌐 Ethical Considerations](#-ethical-considerations)
- [🔐 Cyber Security Considerations](#-cyber-security-considerations)

---

## ⚙️ General Development Risks

| ID | Risk Statement | P (Probability) | I (Impact) | P × I (Risk Score) | Risk Justification |
|----|----------------|------------------|------------|---------------------|---------------------|
| 1  | Lack of experiences and Qualifications In the development team. | 0.9 | 4 | 3.6 | The development team are still learning which may lead to delays, deployment, and quality issues.  Furthermore, project period must be completed within 12 weeks which also may increase significant impacts in this project. |
| 2  | The software implementation may not align with stakeholders’ expectations in terms of the functionality, including visualization design, performance, security, usability, algorithms analysis, documentation, and interactivity. | 0.8 | 4 | 3.2 | As the development team still in the learning phase, they may misinterpret stakeholder expectations, as the stakeholders provided a new feedback in the meetings that have been conducted at the biggening of the project, these feedback may also leads to delays, stakeholder dissatisfaction or the need for modification. |
| 3 | Lack of communication or availability between the team may delay progress | 0.4 | 3 | 1.2 | Team members may have personal commitments or academic responsibilities, which could delay project delivery. Setting up a clear communication channel and distributing work equally within the team is critical to ensure the project is delivered on time |
| 4 | The project may lack testing which could affect the quality of the project. | 0.7 | 3 | 2.1 | Features such as filters, visualizations, and data export may not be fully tested due to time limitations or lack of proper testing practices, which could result in bugs, crashes, or unexpected behavior. |

---

## 🌐 Ethical Considerations

| ID | Risk Statement | P (Probability) | I (Impact) | P × I (Risk Score) | Risk Justification |
|----|----------------|------------------|------------|---------------------|---------------------|
| 5  | Ethical considerations: **Data Privacy** – User data may be vulnerable to unauthorized access or mishandling due to inadequate local privacy protections. | 0.3 | 3 | 0.9 | Even though the software runs locally, data such as user inputs, results, or logs may still be vulnerable to unauthorized access if stored unencrypted or without access controls. Ethically, such data needs to be protected to preserve user privacy and maintain user trust. |
| 6  | Ethical considerations: **Transparency** – The system's workflow and decision-making process may not be clear to the user. | 0.6 | 3 | 1.8 | Inadequate documentation, unclear user interfaces, or unclear processes may prevent users from understanding how to interact with the system or trust it. Ethically, transparency is crucial for building trust, accountability, and helping users clearly understand what the system does before they agree to use it. |
| 7  | Ethical considerations: **Accountability** – The developers may ignore or fail to fix critical issues that exist in the software, such as unclear features, bugs or the software is user-unfriendly. | 0.4 | 3 | 1.2 | While the software is running locally and built by students, some critical problems may go unnoticed or may not be fixed in time due to the developers’ limited experience. Ethically, the developers are responsible for delivering high-quality work, understanding how the software functions, and responding to stakeholder feedback to ensure it meets their expectations. |
| 8 | Ethical considerations: **Fairness** –  The system may unintentionally introduce bias toward specific features and algorithms. | 0.4 | 9 | 1.2 | The visualization of algorithm comparison, for example always showing the best performer may affect user perception. **Algorithm fairness** is important for fair evaluation and accurate analysis. |
| 10 | Ethical considerations: **Informed Consent and Misinterpretation** – Visualizations or statistical results may be misinterpreted by the users. | 0.6 | 3 | 1.8 | The system may not provide clear explanations, warnings, or basic information, which can significantly impact the user's understanding and potentially affect the quality of research. |
| 11 | Ethical considerations: **Inclusivity** – The dashboard may lack accessibility features for users with special needs. | 0.4 | 2 | 0.8 | Users with special needs may struggle to use the software or dashboard if it relies mainly on colors to show important information and does not support text, patterns, or screen readers. |
| 12 | Ethical considerations: **Sustainability** – The system or dashboard may consume high amounts of computational resources | 0.3 | 2 | 0.6 | Developers need to use efficient algorithms or features that have better performance for visualizations to avoid negative impacts on CPU, memory, and energy consumption. |

---

## 🔐 Cyber Security Considerations

| ID | Risk Statement | P (Probability) | I (Impact) | P × I (Risk Score) | Risk Justification |
|----|----------------|------------------|------------|---------------------|---------------------|
| 13  | Ethical considerations: **Data Privacy** – User data may be vulnerable to unauthorized access or mishandling due to inadequate local privacy protections. | 0.3 | 3 | 0.9 | Even though the software runs locally, data such as user inputs, results, or logs may still be vulnerable to unauthorized access if stored unencrypted or without access controls. Ethically, such data needs to be protected to preserve user privacy and maintain user trust. |
| 14  | The data exportation file may not share or store precariously. | 0.6 | 2 | 1.2 | Exported files of visualizations or analysis results may be saved by users in insecure locations. Furthermore, they may be transmitted through unprotected channels, which could compromise the data. The system needs to follow secure export practices. |
| 15  | While the Application Programming Interface (API)  is running locally, it may be accessed without protection. | 0.3 | 3 | 0.9 | If the dashboard of the software contains sensitive data like exporting or modifying data that can be accessed without permission, such as not adding passwords, other applications or users on the same computer can access them, which may significantly affect the security of this sensitive data. |
| 16  | If the local device does not contain a login password, user accounts, locking the screen or session timeout, could lead to misuse. | 0.4 | 3 | 1.2 | Other users may accidentally access or edit open projects if shared environments, like university labs, lack login passwords, user accounts, screen locking, or session timeouts. This could lead to misuse and may also have a potential impact on the project. |
| 17  | Cached or temporary data may still can be accessed after session ends. | 0.3 | 2 | 0.6 | Functions like optimization and caching (PERF-1) may temporarily store results locally. If they are not secured or cleared, it could negatively impact the security of sensitive data, which may be accessible to other users. |
| 18 | The input validation of the system may not be accurate, may not work correctly, or may not sanitize the user input, which can cause harmful impacts on the data or the system. | 0.5 | 3 | 1.5 | EF-1 features allow users to enter specific filters or metrics. If the system does not perform proper validation, the user may crash the system or cause unexpected results. Developers need to follow secure coding practices. |
| 19 | The third-party packages may contain bugs or security vulnerabilities. | 0.2 | 2 | 0.4 | The system relies on packages like instancespace. If these packages are not updated or monitored, they could cause security issues even in local environments. |
