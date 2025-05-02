## OP-RedBack

Welcome to the official repository for the **OP-RedBack**. This project aims to replace the existing MATLAB-based system used by OPTIMA researchers with an interactive, modular, and user-friendly Python-based dashboard.

---

##  Project Overview

The goal of this project is to create a dashboard that visualizes and supports the Instance Space Analysis package workflow using Python technologies like Streamlit. It enables researchers to:

- Visualize and analyze algorithm performance across different stages
- Interact with projection data and adjust parameters iteratively
- Explore and filter datasets without deep technical overhead

---

## 📦 Prerequisites

Ensure the following tools are installed:

- [Python 3.12+](https://www.python.org/downloads/)
- `pip` (comes with Python)
- Virtual environment tool (`venv` or `conda` recommended)

---

## 🚀 How to Run

#### 1. Clone the repository

```bash
git clone https://github.com/FEIT-COMP90082-2025-SM1/OP-RedBack.git
cd OP-RedBack
```

#### 2. Set up and activate the virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the Streamlit app

```bash
streamlit run src/app.py
```

> Then open the URL in your browser (usually at `http://localhost:8501`).

---

## Repository Structure

```bash
├── docs/                # Documentation files (copied from the Wiki)
├── src/                 # Source code (to be added in Sprint 2)
└── README.md            # Overview, repo layout, guidelines, and changelog
```

- `docs/` contains project documentation such as user stories, requirements, and the motivational model.
- `src/` will contain Python source code, visualizations, and Streamlit app components in Sprint 2.
- `README.md` is updated across sprints and contains this project summary, structure, workflow, and changelog.

---

## GitHub Workflow

### 1. Branching Strategy  
We follow a **GitFlow**-based branching strategy:

| Branch                 | Purpose                                 |
|------------------------|-----------------------------------------|
| `master`               | Stable releases                         |
| `develop`              | Main development branch                 |
| `feature/<name>`       | New features (branched off `develop`)   |
| `release/<sprint>`     | Preparation for a sprint release        |
| `hotfix/<issue>`       | Urgent fixes for `master` or `release`  |
| `test/<name>`          | Temporary branches for integration testing or deployment verification |

---
### 2. Sprint Workflow

#### a. Sprint Kick-off & Feature Branching  
- **When**: As soon as the sprint backlog is finalized.  
- **How**:  
  ```bash
  git checkout develop
  git pull origin develop
  git checkout -b feature/<descriptive-name>
  ```  
- **Naming**: Use descriptive names, e.g. feature/user-authentication.

#### b. **Ongoing Integration & Pull Requests**  
   - Developers regularly commit and push to their feature branches.  
   - When a feature is “ready for review,” open a Pull Request targeting `develop`.  
   - The team conducts code review, addresses feedback, and only after approval and passing CI checks is the branch merged back into `develop`.  
   - **Never** merge directly into `develop` without a PR.

#### c. **Daily Syncs & Conflict Resolution**  
   - To minimize drift, each morning (or before starting new work) pull the latest `develop` into your feature branch:  
     ```bash
     git fetch origin  
     git checkout feature/<name>  
     git rebase origin/develop  
     ```  
   - Resolve any conflicts immediately, then force-push the rebased branch (`git push --force-with-lease`).

#### d. **Release Branch Creation**  
   - When the sprint’s scope is complete (all targeted features merged into `develop`), we cut a release branch:  
     ```bash
     git checkout develop  
     git pull origin develop  
     git checkout -b release/<sprint-number>  
     ```  
   - This branch is used for final QA, documentation updates, and any last-minute adjustments (e.g., version bump, changelog entry).

#### e. **Release Stabilization & Hotfixes**  
   - **Only bug fixes or release-specific tweaks** are merged into `release/<sprint>`. All new features must wait for the next sprint.  
   - If a critical bug is discovered _after_ the release branch is cut, create a `hotfix/<issue>` branch from `release/<sprint>` (or `master` if already released), fix the issue, then merge back into both `release/<sprint>` and `develop` once verified.

#### f. **Final Release Merge**  
   - Once QA signs off on `release/<sprint>`, merge it into `master` (triggering the official deploy/tag) and back into `develop` to ensure that any final tweaks are not lost:  
     ```bash
     git checkout master  
     git merge --no-ff release/<sprint>  
     git tag v<version>  
     git checkout develop  
     git merge --no-ff release/<sprint>  
     ```  
   - Delete the `release/<sprint>` branch after merging.

### 3. Commit Message Conventions

 Commit messages follow this format:  
`<type>(<scope>): <short description>`

| Type      | Description                            |
|-----------|----------------------------------------|
| feat      | New feature                            |
| fix       | Bug fix                                |
| docs      | Documentation only                     |
| style     | Formatting, no logic change            |
| refactor  | Code restructuring without behavior    |
| test      | Adding or updating tests               |
| chore     | Maintenance tasks                      |

---

##  Sprint 1&2 Deliverables

All wiki pages have been exported to `docs/` as Markdown files.

View the full documentation index here:  
[Documentation Index](./docs/README.md)

---

#Sprint 1 Changelog

✅ **Defined the client (OPTIMA)**, documented project motivations, clarified realistic scope, and structured background in an error-free format.  
✅ **Gathered both functional and non-functional requirements**, drafted clear user stories grouped by epics, and aligned them with the project scope.
✅ **Outlined a clear Motivational Model**, including stakeholder needs, functional/quality goals, and emotional goals to guide dashboard design.  
✅ **Created a complete user story map**, outlined key workflows and dependencies, and documented planning for Sprint 2 in an accessible format.  
✅ **Set up the GitHub repository** following the required structure (`docs/`, `src/`, `README.md`) and established GitFlow with naming conventions and commit message rules.  
✅ **Organised the team workspace** using GitHub Project boards and Slack, ensuring all tools are actively maintained for collaboration.  
✅ **Validated user stories and prototype with industry partner**, recorded a 3–5 min walkthrough, incorporated feedback, and documented key takeaways.  

Sprint 2 Changelog

# 🛠 Sprint 2 Changelog

✅ Deployed the preprocessing stage using Streamlit, allowing users to upload files and initiate instance space analysis.  
✅ Completed a structured product demonstration showcasing the preprocessing flow and partial dashboard functionality.  
✅ Conducted structured code reviews using GitHub Actions and documented participants, reviewed files, key issues, and rationale in the GitHub Wiki.  
✅ Explained how AI assisted during the review process and reflected on how AI-driven suggestions were evaluated, accepted, or dismissed with reasoning.  
✅ Fully addressed Sprint 1 feedback across user stories, GitHub board, planning documentation, and validation deliverables.  
✅ Integrated industry partner goals and background into project planning to align development priorities with stakeholder needs.  
✅ Held Sprint 2 review and planning sessions, documented partner feedback, reflected on Sprint 1 outcomes, and outlined Sprint 3 scope.  
✅ Maintained Agile task tracking using GitHub Projects, including well-defined, estimated, and assigned tasks actively moved across workflow lanes.  
✅ Created and documented workflow dependency diagrams and linked them to team roles and backlog refinement.  
✅ Documented handover processes and estimation logic in the GitHub Wiki, along with the task board and story maps.  
✅ Defined and applied quality assurance standards for contributions and collaborative development, including secure coding practices.  
✅ Assessed ethical risks (privacy, transparency, inclusivity, sustainability) and documented mitigation strategies in the GitHub Wiki.  
✅ Integrated ethical guidelines into the product development cycle and updated them to reflect current project direction.  
✅ Identified cybersecurity risks and documented secure coding practices, third-party package risks, and potential threat surfaces.  
✅ Outlined planned authentication and access control strategies as part of future deployment considerations.  
✅ Published all sprint ceremony minutes, including retrospective, review, planning, and technical walkthroughs, with summaries and team input.  
✅ Updated GitHub Wiki with review logs, planning artifacts, architecture diagrams, and documentation links.  
✅ Completed documentation of stakeholder impact, dependencies, roles, and quality checkpoints to prepare for Sprint 3 handover and continuity.

