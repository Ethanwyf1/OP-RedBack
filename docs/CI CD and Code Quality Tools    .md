## CI/CD and Code Quality Tools

### Purpose  
Ensure consistent, maintainable code in the Instancespace Streamlit project by automating code review using ChatGPT and linting tools. Testing is done manually by developers.

---

### Active CI Workflows (via GitHub Actions)

All workflows are stored in `.github/workflows/` and run on pull request events.

#### ✅ ChatGPT PR Review (`pr-chat-review.yaml`)
- Uses `agogear/chatgpt-pr-review`
- Triggers on every PR (`opened`, `synchronize`)
- Reviews all `.py` files changed in the PR
- Posts suggestions directly as comments (e.g., missing docstrings, logic clarity, redundant code)
- Requires `OPENAI_API_KEY` and `GITHUB_TOKEN` as secrets

#### ✅ Linting (`lint.yml`)
- Uses `flake8` to enforce code cleanliness
- Checks:
  - PEP8 formatting
  - Unused imports or variables
  - Long lines or indentation issues
- Optional — can be enabled or disabled based on project needs

---

### Manual Testing Process

At this stage, **no automated tests (like pytest)** are configured.  
Developers are expected to:

- Manually run the app and test core functionalities after changes
- Confirm no runtime errors are introduced
- Visually check Streamlit outputs as part of feature delivery

---

### Developer Checklist (Each PR)

- [ ] Code follows naming and structural conventions  
- [ ] ChatGPT feedback is reviewed and resolved  
- [ ] Linting issues are fixed (if enabled)  
- [ ] Feature is manually tested in Streamlit  
- [ ] PR includes a clear summary and module-specific context
