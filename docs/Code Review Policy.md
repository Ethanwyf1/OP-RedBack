## 🔧 Code Review Policy

### Purpose
To ensure that all stage modules in the Instancespace Streamlit dashboard are reviewed for clarity, modularity, and research accuracy using a consistent, AI-augmented workflow.

---

### Core Principles

- Every new stage (`*.py`) is reviewed individually through a pull request (PR)
- ChatGPT is used as the **first-pass reviewer** before any human review
- Final production code is merged only once via the `dev → main` PR after all feedback is addressed

---

### Pull Request Review Workflow

#### 1. Create Isolated PRs per Stage
- Open one PR per module (e.g., `preprocessing.py`, `prelim.py`, `sifted.py`)
- Base each PR on the latest `main`
- Do **not merge** these PRs — they are used only for triggering ChatGPT review

#### 2. Run ChatGPT Review
- GitHub Action runs ChatGPT automated review
- Feedback is posted as comments on the PR
- Developers summarize and respond to suggestions (accepted/rejected)

#### 3. Integrate into Development Branch
- All final changes are committed to the `dev` branch
- Each module is updated in `dev` with ChatGPT-reviewed improvements

#### 4. Final Merge to Main
- A single PR is opened from `dev → main`:
  ```
  PR: Merge all reviewed stages into main
  ```
- Requires at least one human review
- Merge conditions:
  - ✅ All ChatGPT feedback resolved
  - ✅ CI passes
  - ✅ No blocking comments

---

## ✅ Pull Request Template

```markdown
### Summary
Implements `<stage>.py`, which handles <brief one-line function>.  
See wiki for full module behavior.

---

### ChatGPT Review Summary
_(Filled after AI review is posted)_

- [ ] ...
- [ ] ...

---

## QA Checklist

- [ ] Title + description provided
- [ ] PR is focused (one module)
- [ ] Code structure followed
- [ ] ChatGPT suggestions handled
- [ ] Reviewer assigned
- [ ] Docs updated (✅ wiki)
```

---

## 🧠 Review Strategy (AI-Human Hybrid)

### Why This Approach

- ✅ Keeps each stage testable in isolation
- ✅ Reduces merge conflicts
- ✅ Allows clean, centralized integration
- ✅ Ensures ChatGPT suggestions are contextual and modular

### ChatGPT’s Role
Used as a static analysis layer to:
- Highlight docstring/comment gaps
- Detect redundant code or unused imports
- Surface structural or naming concerns
- Provide fast, file-scoped feedback

### Developer’s Role
- Interpret ChatGPT suggestions
- Apply or reject with reasoning
- Log outcomes in the final integration branch

