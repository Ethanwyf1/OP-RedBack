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
## 📦 Project Releases

Sprint-based releases are published on GitHub to track deliverables and project progress.  
You can find all release packages, changelogs, and demonstration links at:

🔗 [GitHub Releases – OP-RedBack](https://github.com/FEIT-COMP90082-2025-SM1/OP-RedBack/releases)

Latest release:  
🎯 [Sprint 2](https://github.com/FEIT-COMP90082-2025-SM1/OP-RedBack/releases/tag/COMP90082_2025_SM1_OP-RedBack_RL_SPRINT2)

---

## Sprint 1 Changelog

✅ **Defined the client (OPTIMA)**, documented project motivations, clarified realistic scope, and structured background in an error-free format.  
✅ **Gathered both functional and non-functional requirements**, drafted clear user stories grouped by epics, and aligned them with the project scope.
✅ **Outlined a clear Motivational Model**, including stakeholder needs, functional/quality goals, and emotional goals to guide dashboard design.  
✅ **Created a complete user story map**, outlined key workflows and dependencies, and documented planning for Sprint 2 in an accessible format.  
✅ **Set up the GitHub repository** following the required structure (`docs/`, `src/`, `README.md`) and established GitFlow with naming conventions and commit message rules.  
✅ **Organised the team workspace** using GitHub Project boards and Slack, ensuring all tools are actively maintained for collaboration.  
✅ **Validated user stories and prototype with industry partner**, recorded a 3–5 min walkthrough, incorporated feedback, and documented key takeaways.  


##  Sprint 2 Changelog

✅ Implemented a Streamlit-based preprocessing interface with support for `metadata.csv` uploads, algorithm and feature dropdown filters, real-time visualizations, cache generation, and downloadable outputs.  
✅ Updated the `README.md` to include setup instructions, usage guide, system workflow, and architectural overview.  
✅ Delivered a structured product demonstration showcasing key preprocessing features and progress since Sprint 1.  
✅ Conducted formal code reviews using GitHub Actions; documented participants, reviewed files, identified issues, and addressed feedback.  
✅ Reflected on AI-assisted code reviews, explaining which suggestions were accepted or dismissed and why.  
✅ Incorporated industry partner background and goals into project planning and documentation.  
✅ Completed Sprint 2 review and planning sessions, documented partner feedback, and aligned Sprint 3 scope accordingly.  
✅ Maintained and updated the GitHub Project board with clearly defined, estimated, and assigned tasks across all workflow lanes.  
✅ Documented task dependencies and estimation processes in the GitHub Wiki.  
✅ Outlined and enforced quality assurance practices, including secure coding standards and contribution guidelines.  
✅ Assessed and documented ethical concerns (privacy, inclusivity, transparency, sustainability) and integrated them into the product development process.  
✅ Identified cybersecurity risks, secure API practices, authentication considerations, and third-party package assessments; updated relevant documentation.  
✅ Captured and published minutes for all sprint ceremonies, including planning, review, retrospective, and technical walkthroughs.  
✅ Finalized all Sprint 2 documentation, including team roles, stakeholder impact, workflow dependencies, and handover preparation for Sprint 3.


