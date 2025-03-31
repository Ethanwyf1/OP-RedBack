# OP-RedBack


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



