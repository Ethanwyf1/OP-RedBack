## 🛡️ Risk Mitigation Strategies

This document outlines mitigation strategies for the identified project risks across **General Development Risks**, **Ethical Considerations**, and **Cyber Security Considerations**. Each strategy includes the risk trigger, owner, type of response, and the proposed approach to minimize or prevent the impact.

---

## 📑 Table of Contents

- [General Development Risks](#general-development-risks)
- [Ethical Considerations](#ethical-considerations)
- [Cyber Security Considerations](#cyber-security-considerations)

---

## General Development Risks

| ID | Risk Trigger | Owner | Response Type | Risk Response Strategy |
|----|--------------|-------|----------------|-------------------------|
| 1  | Lack of experience in the development team may lead to delays and issues. | Scrum Master and Developer Lead | Mitigation and Preventive | Prioritize core features early and support the team with resources and peer programming. Regular check-ins help monitor learning progress and address challenges early. |
| 2  | Misunderstanding stakeholder expectations regarding key features or usability. | Product Owner and Scrum Master | Mitigation | Organize weekly meetings with stakeholders to align expectations. Update wireframes and backlog items frequently to reflect feedback. |
| 3  | Communication issues or unavailability among team members. | Scrum Master | Mitigation | Set up a shared communication platform and ensure all members agree on availability hours. Track responsibilities clearly on the task board. |
| 4  | Limited testing time which might result in bugs or feature failures. | QA Lead and Developers | Mitigation and Preventive | Assign testing sessions within each sprint. Use checklists and test cases. Promote unit testing and pair testing among team members. |

---

## Ethical Considerations

| ID | Risk Trigger | Owner | Response Type | Risk Response Strategy |
|----|--------------|-------|----------------|-------------------------|
| 5  | User data may be mishandled or stored unsafely. | QA Lead | Preventive | Keep sensitive data local, avoid storing more than needed, and use encryption where possible. Explain privacy practices to users. |
| 6  | Lack of clarity in how the system makes decisions. | Developer Team | Mitigation | Include tooltips, onboarding notes, and step-by-step guides. The development team will prepare simple walkthroughs and help messages to support user understanding. |
| 7  | Developers may miss critical bugs or unclear sections. | Developer Team | Preventive | Create an issue log for tracking bugs. Allocate time during sprints for debugging. Peer review each other’s work before merging. |
| 8  | Bias in visualizations may affect how results are interpreted. | Developer Lead | Mitigation | Allow multiple visualization modes (e.g., ranked vs raw values). Clarify how performance is calculated. |
| 10 | Users might misread charts or metrics. | Developer & QA | Preventive | Add clear labels and legends on graphs. Include a walkthrough tutorial or example dataset. |
| 11 | Limited accessibility for users with visual impairments or disabilities. | Developer Team | Mitigation | Use proper color contrast, support screen readers, and include text alternatives. Test with accessibility tools. |
| 12 | System consumes too much processing power. | Developers | Preventive | Monitor performance using profiling tools. Allow users to disable heavy visual features and optimize backend processes. |

---

## Cyber Security Considerations

| ID | Risk Trigger | Owner | Response Type | Risk Response Strategy |
|----|--------------|-------|----------------|-------------------------|
| 13 | Sensitive data stored insecurely on local devices. | QA Lead | Preventive | Avoid unnecessary data logging. Use protected folders or encrypted storage. Remind users to clear logs after use. |
| 14 | Users might save or share export files unsafely. | Developer Team | Mitigation | Use temporary file locations. Show a warning message on export suggesting safe file handling. |
| 15 | APIs might be accessible without any local restrictions. | Developer | Mitigation | Add local authentication like tokens or session checking to API endpoints. Log all API usage. |
| 16 | Shared computers without session protection could lead to data leaks. | QA Lead | Preventive | Enable auto-logout or idle timeout. Inform users about securing their devices when working in shared labs. |
| 17 | Cache remains after app closes. | Developer | Mitigation | Clear cache on exit or provide a manual “Clear Cache” option in the UI. Avoid permanent storage. |
| 18 | Poor input validation may allow invalid or harmful inputs. | Developer Team | Preventive | Use validation libraries. Check all user input and sanitize strings before processing. Log invalid attempts for review. |
| 19 | Bugs or issues in third-party packages used in the dashboard. | Developer Team | Mitigation | Monitor dependencies using tools like Dependabot. Keep libraries up to date and check release notes for vulnerabilities. |
