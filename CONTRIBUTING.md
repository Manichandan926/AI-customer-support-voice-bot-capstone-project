# Contributing

Thanks for contributing to the AI customer support voice bot prototype.

## Local setup

Create a virtual environment and install the lightweight runtime dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If you need the optional Groq-backed AI response generator, install:

```bash
pip install -r requirements-llm.txt
export GROQ_API_KEY="your_key_here"
```

## Quality gates

This project protects the main branch with a simple rule: no merge should happen unless the code passes the repository test suite.

Before you commit, run:

```bash
python3 -m unittest discover -v
```

The repository also ships with a Git pre-commit hook that runs the test suite automatically:

```bash
./scripts/install_hooks.sh
```

## Branching and PR flow

- Work on a feature branch.
- Keep changes small and focused.
- Run the test suite before opening a pull request.
- Open the PR against `main`.
- Ensure all required checks are green before merge.

## Standards

- Prefer small, reviewable changes.
- Keep the runtime lightweight; do not add heavy dependencies unless they are clearly required.
- Preserve the modular input/NLU/dialogue/output design.
- Update documentation when behavior or setup changes.
