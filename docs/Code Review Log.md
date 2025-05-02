# Code Review Log

This log captures how the team systematically engaged with AI-assisted code reviews across Sprints. Each review demonstrates deliberate evaluation of ChatGPT feedback, thoughtful incorporation of valuable suggestions, and documented reasoning for deferred changes. The team prioritised maintainability, clarity, and consistency, balancing automation with human judgment.

---

## Pull Request Template

| Field                 | Details                            |
|----------------------|-------------------------------------|
| **PR Title**          | [Enter PR title]                   |
| **Sprint**            | Sprint #                           |
| **Files Reviewed**    | [e.g. src/stages/preprocessing.py] |
| **PR Link**           | [Paste PR URL here]                |
| **Date Reviewed**     | [YYYY-MM-DD]                       |
| **Author**            | [Full Name]                        |
| **Human Reviewer(s)** | [List names]                       |
| **QA Lead Involved**  | [Name if applicable]               |
| **Issues Identified** | [Count]                            |
| **Issues Addressed**  | [Count]                            |
| **Issues Deferred**   | [Count]                            |

### Summary of Changes  
Brief description of the purpose and scope of the pull request, including why the reviewed file/module was significant in the pipeline.

### ChatGPT Suggestions  
- [ ] Summarize actionable AI feedback (e.g., structure, naming, clarity)  
- [ ] List any flagged issues related to scalability, duplication, or performance  

### Actions Taken  
- Detail what feedback was implemented and why  
- List suggestions deferred or rejected, with brief rationale  

### AI Engagement & Reflection  
Summarize how the team analyzed each AI comment, weighed its relevance, and applied changes where they added real value. Clearly distinguish between suggestions applied versus those consciously postponed.

### Lessons Learned  
Document any emerging standards, good practices, or team insights that resulted from the review process.

---

## Reviewed Entry: Preprocessing Stage

| Field                 | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **PR Title**          | PR: Add preprocessing stage module                                      |
| **Sprint**            | Sprint 2                                                                 |
| **Files Reviewed**    | `src/stages/preprocessing.py` – entry point for data preparation        |
| **PR Link**           | https://github.com/FEIT-COMP90082-2025-SM1/OP-RedBack/pull/116                                          |
| **Date Reviewed**     | 2025-05-01                                                               |
| **Author**            | Abdulrahman Alaql                                                       |
| **Human Reviewer(s)** | ZaneX-994                                                               |
| **QA Lead Involved**  | Abdulrahman Alaql                                                       |
| **Issues Identified** | 4                                                                       |
| **Issues Addressed**  | 3                                                                       |
| **Issues Deferred**   | 1                                                                       |

### Summary of Changes  
Initial implementation of the preprocessing stage. Includes metadata input handling, dynamic feature/algorithm selection, Streamlit interface, data caching, and basic visualizations.

### ChatGPT Suggestions  
1. Rename `show()` to a more descriptive name  
2. Add comments to clarify selection and visualization blocks  
3. Split complex dictionary lines for readability  
4. Flag potential memory issues with large visualizations  

### Actions Taken  
- Commenting was improved across all core logic  
- Complex initialisations were reformatted for clarity  
- The `show()` function was kept for structural consistency with other modules  
- Memory concerns were acknowledged for future testing, not urgent for current scope  

### AI Engagement & Reflection  
The team evaluated ChatGPT’s feedback in detail. Suggestions that improved readability were adopted immediately. Naming consistency across pipeline stages was prioritized over renaming a single function. The team demonstrated balanced decision-making between adopting improvements and preserving system-wide uniformity.

### Lessons Learned  
Clear inline documentation helps reviewers onboard faster. Memory-conscious design will be revisited post-integration. The value of AI is most evident when paired with team-wide consistency goals.

---

## Reviewed Entry: Prelim Stage

| Field                 | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **PR Title**          | PR: Add prelim stage module                                             |
| **Sprint**            | Sprint 2                                                                 |
| **Files Reviewed**    | `src/stages/prelim.py` – performance analysis and normalization logic    |
| **PR Link**           | https://github.com/FEIT-COMP90082-2025-SM1/OP-RedBack/pull/117                                        |
| **Date Reviewed**     | 2025-05-01                                                               |
| **Author**            | Abdulrahman Alaql                                                       |
| **Human Reviewer(s)** | ArulananthamAnujan                                                      |
| **QA Lead Involved**  | Abdulrahman Alaql                                                       |
| **Issues Identified** | 5                                                                       |
| **Issues Addressed**  | 4                                                                       |
| **Issues Deferred**   | 1                                                                       |

### Summary of Changes  
Implements logic for performance thresholding, data transformation, algorithm ranking, statistical summaries, and exportable output for downstream stages.

### ChatGPT Suggestions  
1. Expand docstrings and add detailed inline comments  
2. Reorganize long lines in visualizations  
3. Refactor repeated plotting logic  
4. Validate widget inputs explicitly  
5. Guarantee temporary file cleanup during failures  

### Actions Taken  
- Added comments and docstrings in all major functions  
- Visualization code was cleaned up for clarity  
- Refactoring plotting logic into helpers was noted but deferred to avoid scope creep  
- Input validation was strengthened with type-checked defaults  
- Cleanup logic was verified and improved  

### AI Engagement & Reflection  
Each suggestion was critically reviewed. Readability and safety improvements were implemented. Larger refactors were documented and postponed for a coordinated design pass post-feature freeze. The team prioritized maintainable growth over premature optimization.

### Lessons Learned  
The review reinforced the need for reusable plotting utilities. AI helped highlight hidden logic duplication. Manual validation of external outputs (e.g., file handling) remains a necessary complement to automated reviews.

---

## Reviewed Entry: Sifted Stage

| Field                 | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **PR Title**          | PR: Add sifted stage module                                             |
| **Sprint**            | Sprint 2                                                                 |
| **Files Reviewed**    | `src/stages/sifted.py` – clustering, GA-based selection, final feature curation |
| **PR Link**           | https://github.com/FEIT-COMP90082-2025-SM1/OP-RedBack/pull/118                                          |
| **Date Reviewed**     | 2025-05-01                                                               |
| **Author**            | Abdulrahman Alaql                                                       |
| **Human Reviewer(s)** | ZaneX-994                                                               |
| **QA Lead Involved**  | Abdulrahman Alaql                                                       |
| **Issues Identified** | 5                                                                       |
| **Issues Addressed**  | 3                                                                       |
| **Issues Deferred**   | 2                                                                       |

### Summary of Changes  
Implements the SIFTED stage, handling correlation-based feature selection, genetic algorithm tuning, silhouette scoring, PCA, and instance filtering. Includes complex configuration UI and multi-step visual outputs.

### ChatGPT Suggestions  
1. Rename variables like `flag`, `rho`, and `k` for clarity  
2. Collapse repeated plot-download logic into helper functions  
3. Improve docstring coverage in visual analysis code  
4. Ensure widget input constraints are defined  
5. Clarify section headers and structure for maintainability  

### Actions Taken  
- Comments were added for parameter tuning logic  
- Input widgets were updated with consistent constraints  
- Code structure and spacing were improved for readability  
- Variable renaming and plotting helpers were postponed to a later consistency pass  

### AI Engagement & Reflection  
The team reviewed all five points and applied those that clearly improved UX and code clarity. Variable renaming was deferred due to active discussion on project-wide naming standards. Repetitive logic was acknowledged as a refactor target but was deprioritized in favor of stability before merge.

### Lessons Learned  
AI-assisted reviews raised valid structure and naming concerns. Still, the team emphasized cohesion across modules, demonstrating that AI suggestions are most useful when filtered through a broader architectural lens.

---
