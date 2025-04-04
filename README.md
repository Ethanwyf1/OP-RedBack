# OP-RedBack

## 📁 Project Documentation (Sprint 1)

All wiki pages have been copied to the [`docs/`](./docs/) folder as Markdown files.

To help navigate them, see the [Documentation Index](./docs/README.md).

### Branching Strategy: GitFlow

This branching strategy consists of the following branches:


- Master 
- Develop
- Feature- to develop new features that branches off the develop branch 
- Release- help prepare a new production release; usually branched from the develop branch and must be merged back to both develop and master
- Hotfix- also helps prepare for a release but unlike release branches, hotfix branches arise from a bug that has been discovered and must be resolved; it enables developers to keep working on their own changes on the develop branch while the bug is being fixed.

### 🚀 Commit Guidlines

Each commit message should follow this format:

`<type>(<scope>): <short description>`


---

#### **1️⃣ Commit Type**

| Type       | Purpose |
|------------|--------------------------------------------------------------|
| `feat`     | Introduces a new feature (e.g., UI component, caching system). |
| `fix`      | Fixes a bug or issue (e.g., broken visualization, security bug). |
| `refactor` | Improves existing code without changing functionality. |
| `perf`     | Performance improvements (e.g., optimizing rendering speed). |
| `docs`     | Documentation updates (e.g., README, API reference). |
| `style`    | Code style changes (e.g., formatting, linting, no logic change). |
| `test`     | Adding or modifying tests (e.g., unit tests for visualization). |
| `chore`    | Maintenance tasks (e.g., dependency updates, build scripts). |

---

#### **2️⃣ Scope (Which part of the project does it affect?)**

Examples of scopes to use within commit messages:

- `ui` – User interface components  
- `backend` – Data handling, security, caching  
- `viz` – Data visualization, plots, graphs  
- `docs` – Documentation updates  
- `auth` – Authentication and access control  
- `tests` – Testing framework updates 

---

#### **3️⃣ Examples of Commit Messages**

### ✅ Good commits:

feat(viz): add interactive zoom for instance space plots

### 🏁 Sprint 1 Changelog (Planning, Setup & Documentation)

- ✅ **Defined the client (OPTIMA)**, documented project motivations, clarified realistic scope, and structured background in an error-free format.  
- ✅ **Gathered both functional and non-functional requirements**, drafted clear user stories grouped by epics, and aligned them with the project scope.  
- ✅ **Created a complete user story map**, outlined key workflows and dependencies, and documented planning for Sprint 2 in an accessible format.  
- ✅ **Set up the GitHub repository** following the required structure (`docs/`, `src/`, `README.md`) and established GitFlow with naming conventions and commit message rules.  
- ✅ **Organised the team workspace** using GitHub Project boards and Slack, ensuring all tools are actively maintained for collaboration.  
- ✅ **Validated user stories and prototype with industry partner**, recorded a 3–5 min walkthrough, incorporated feedback, and documented key takeaways.  





