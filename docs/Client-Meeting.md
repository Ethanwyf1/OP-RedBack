# **On this page, you can find all notes and agendas from meetings with OPTIMA's clients.**

# **Initial Meeting agendas**


**Participants:** Dr. Mario, Team Members, Ashley

The meeting was held on March 19, 2025.




**1. Welcome & Introductions (5-10 min)**

> Brief self-introductions of all attendees
> Overview of our team’s background and expertise
> Client introductions and roles.

**2. Project Overview (20-30 min)**

> Understanding the client's business and goals
> High-level discussion of the project scope
> Key problems the project aims to solve.

**3. Requirements Gathering (10-20 min)**

> Discuss client expectations and specific needs
> Identify functional and non-functional requirements
> Clarify target users and key use cases
> Explore any existing systems or integrations.

**4. Project Constraints & Considerations (5-10 min)**

> Timeline expectations and key milestones
> Budget constraints (if applicable)
> Technical preferences or limitations
> Compliance, security, or regulatory requirements.

**5. Next Steps & Action Items (10 min)**

> Summarize key takeaways from the meeting
> Define immediate next steps and responsibilities
> Schedule follow-up meetings or check-ins.


**Questions：**

> 1. Who are the target users? Researchers, analysts, or business users?
> 2. What key insights or metrics should the dashboard display?
> 3. Do you have any existing references or design preferences?
> 4. Do you have any preferences for the technology stack? (Dash, Streamlit, Plotly, React?
> 5. What is the main goal of this visualization dashboard?
> 6. What are the main functions and content that this dashboard needs to fulfill?
> 7. Algorithmic weaknesses
> 8. Algorithm Comparison (Chart?)
> 9. What improvements do you expect from the new Python-based dashboard?
> 10. Who will be the main point of contact for feedback and approvals?
> 11. Do you have a preferred tool for tracking progress
> 12. What are the biggest risks or challenges you foresee in this project?
> 13. What would success look like for this project from your perspective?
> 14. How frequently would you like updates on progress?



# **Initial Meeting minutes**

**Date:** 19/03/2025

**Participants:** Dr. Mario, Team Members, Ashley

**Discussion Topic:** Understanding the requirements of the Matilda project and the team's tasks


**1. Project Background**

> * The primary(target) users: researchers and scientists in computer science and applied mathematics (around 200 people).

> * **User priorities:**

> > * They do not want to spend time building and maintaining a dashboard but are instead focused on understanding how their algorithms work.

> > * The project currently follows a modular approach to data processing to improve visualization consistency.

> > * The previous implementation was monolithic, meaning that any modification impacted the entire process, leading to inefficiencies.

> > * The current goal is to leverage the modular structure to enhance visualization, making the data more intuitive and understandable for users.

**2. Key Tasks**

> * **Understanding the Existing Data Structure and Workflow**

> * Study the structure and characteristics of the data, which primarily includes:

> > *  Instances of optimization problems such as the Traveling Salesman Problem (TSP).

> > *  Feature data of these instances, which describe the complexity of the problem.

> > *  The goal is to analyze this data without needing to understand the optimization problem itself.

> * **Data processing flow (example):**

> > *  ``Preprocess → Feature Selection → Machine Learning (ML) → Output``

> * **Designing the Visualization Interface**

> >  *  **Explore the preferred data visualization methods for users, such as:**

> > > * Interactive visualization (e.g., hovering over data points to view statistics).

> > > * Main visualization formats, such as tables, scatter plots, and heatmaps.

Example (This is an example of the final output visualization. The visual representations at different stages may vary—they could be tables, numerical values, or other types of charts)：

<img width="849" alt="Screenshot 2025-03-23 at 11 11 13 PM" src="https://github.com/user-attachments/assets/4f06a515-dc2c-45cb-9e42-6787d19ad19e" />


> > * The team needs to design the UI, confirm the plan with Dr. Mario, and then proceed with development.

> **3. Choosing the Technology Stack**

> > * No predefined frontend technology stack; Dr. Mario has given the team freedom to choose the technologies.

> > * However, the selection should consider:

> > > * Future maintainability.

> > > * Integration difficulty within the overall project.

> > > * Whether it allows for easy modifications by other teams in the future.

> **4. Running the Existing Code to Understand the Workflow**

> > * The team needs to execute the full workflow to understand the different stages of data processing and determine which stages require visualization support.

> > * The project is built using Python and relies on Poetry for environment management.

> > * Documentation can be generated using Pdoc with the command: `pdoc instancespace`

<img width="1118" alt="Screenshot 2025-03-23 at 11 25 31 PM" src="https://github.com/user-attachments/assets/7efdf653-4151-41fd-8bd4-78c0ae2fb6c2" />


> **5. Project Plan**

> > **Week 3-4:**

> > > * Run the existing code, understand the workflow, and analyze the data structure.

> > > * Identify which stages require visualization support.

> > > * Design an initial visualization plan and discuss it with Dr. Mario.

> > **Week 5-6:**

> > > * Finalize the UI design, select the technology stack, and report the reasoning behind the choice.

> > > * Begin developing the dashboard after receiving approval from Dr. Mario.

> > **Week 7-11:**

> > > * Gradually implement the dashboard and test its functionality.

> > > * Make adjustments based on feedback and prepare the final version.

> > **Additional Work (If Time Permits):**

> > > * The Dashboard is currently running locally only and may be considered for integration with production environments in the future.

**6. Key Insights & Work Materials from Mario’s Email**

> **1. Reference Materials**

> > * **Matilda Website ([matilda.unimelb.edu.au](https://matilda.unimelb.edu.au/))**


> > > * The primary reference point for understanding the functionality of the current solution.

> > > * Helps in determining what features exist and what improvements are needed.

> > * **pyInstanceSpace Package ([GitHub Repository](https://github.com/andremun/pyInstanceSpace))**


> > > * Main Python package handling the analysis pipeline.

> > > * In some documentation, this package may be referred to as "Matilda".

> > * **Research Paper ([arXiv: 2501.16646](https://arxiv.org/abs/2501.16646))**

> > > * Provides an overview of the package's functionality and the underlying methodology.

> **2. Example & Inspiration for GUI**

> > InstanceSpace-streamlit ([GitHub Repository](https://github.com/vivekkatial/InstanceSpace-streamlit))

> > > * A limited GUI example that visualizes only the final results.

> > > * Can be used as a starting point for designing the new dashboard.

> **3. Optional Work (If Time Permits)**

> > * **Database Integration**

> > > * Consider integrating database functionalities developed by the mentor or from:

> > > > * ez-experimentr Package ([GitHub Repository](https://github.com/vivekkatial/ez-experimentr))

> > > > * This package offers additional data management functionalities.