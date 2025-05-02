# Code Quality Tools & Best Practices

## Purpose

This page outlines the tools, structure, and workflows used to ensure clean, modular, and maintainable code in the Instancespace Streamlit project. It reflects the current AI-assisted review process, simplified CI, and manual testing strategy adopted during development.

---

## Tools in Use

| Tool              | Purpose                                  | How It's Used                                |
|-------------------|-------------------------------------------|----------------------------------------------|
| `flake8`          | Enforces Python code style (PEP8)         | Run locally or via optional GitHub Action    |
| `ChatGPT PR Review` | AI-driven feedback on each pull request | Auto-runs via GitHub Actions on every PR     |
| `GitHub Actions`  | Automates reviews and optional linting    | Executes workflows for PR review and QA      |

---

## Folder Structure Overview

| Path                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `.github/workflows/`   | Contains CI workflows (ChatGPT review, optional lint)                       |
| `src/`                 | Core application logic and Streamlit modules                               |
| `cache/`               | Stores temporary outputs and saved stage results                           |
| `docs/`                | Markdown documentation and wiki exports                                     |
| `external/`            | Datasets or third-party helpers (not team-authored)                         |
| `scripts/`             | Setup scripts or optional development tools                                 |
| `.flake8`              | Linter configuration                                                        |
| `.gitignore`           | Files/folders excluded from version control                                 |
| `README.md`            | Overview, setup steps, contribution notes                                   |
| `requirements.txt`     | Python dependencies                                                         |

### `.flake8` config used
```ini
[flake8]
max-line-length = 100
exclude = venv, __pycache__, docs
```

To run locally:
```bash
flake8 src/
```

---

## GitHub Actions Workflows

Workflows are triggered on all pull requests. Current active workflows:

| Workflow File         | Description                               |
|------------------------|-------------------------------------------|
| `pr-chat-review.yaml` | Runs ChatGPT code review for changed `.py` files |
| `lint.yml` (optional) | Runs `flake8` on staged Python files       |

---

## Developer Workflow (Per PR)

- [ ] Use meaningful PR title and summary  
- [ ] Push only one module per PR (e.g. `prelim.py`, `sifted.py`)  
- [ ] ChatGPT will review the code automatically  
- [ ] Resolve or respond to all ChatGPT comments  
- [ ] Manually test changes using the Streamlit interface  
- [ ] Merge improvements into `dev` — final merge to `main` happens once all modules are ready

---

## Tips for Clean Code

- Keep each module self-contained and readable  
- Write small, focused functions  
- Use docstrings for all public-facing code  
- Avoid copy-pasting logic across stages — abstract when possible  
- Comment on any workaround, placeholder, or temp fix you add  
