# Contributing Guidelines for Dropzilla v4

To maintain a clean, readable, and manageable codebase, all contributions (especially from Codex agents) must adhere to the following guidelines.

## Pull Requests (PRs)

1.  **One PR Per Task:** Each Pull Request should address a single, focused task that corresponds to a specific GitHub Issue. Do not bundle multiple features or fixes into one PR.

2.  **Clear Titles:** PR titles should follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps in generating automated changelogs and makes the commit history easy to read.
    - `feat:` for a new feature.
    - `fix:` for a bug fix.
    - `refactor:` for code changes that neither fix a bug nor add a feature.
    - `docs:` for documentation changes.
    - `test:` for adding or improving tests.
    - `chore:` for build process or auxiliary tool changes.

    *Example: `feat(models): Implement Bayesian Optimization with Hyperopt`*

3.  **Detailed Descriptions:** The PR description must clearly explain *what* was changed and *why*. It should reference the GitHub Issue it resolves (e.g., `Resolves #5`).

4.  **CI Must Pass:** Do not request a review until the Continuous Integration (CI) checks (running `pytest` and `mypy`) have passed successfully. A green checkmark is required.

5.  **Draft PRs:** For work-in-progress, open a **Draft Pull Request**. This signals that the PR is not yet ready for review and prevents accidental merging. Convert it to a ready PR only when all work is complete and CI is passing.

## Code Style

- All code must be formatted according to `black` standards (this will be added to CI later).
- All new functions and classes must have clear docstrings explaining their purpose, arguments, and return values.
- Type hints are required for all function signatures.

Adhering to these guidelines will ensure a smooth and efficient development process for Dropzilla v4.
