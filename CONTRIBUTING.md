# Contributing

Thanks for contributing to the AI customer support voice bot.

## Local setup

Create a virtual environment and install the runtime dependencies (all
CPU-only; nothing pulls in PyTorch or a GPU toolkit):

```bash
python3 -m venv venv
. venv/bin/activate            # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The cloud AI fallback is optional. To enable it, copy `.env.example` to
`.env` and fill in any of `GROQ_API_KEY`, `GEMINI_API_KEY` or
`ANTHROPIC_API_KEY`. Never commit `.env`.

## Quality gates

This project protects the main branch with a simple rule: no merge should happen unless the code passes the repository test suite.

Before you commit, run:

```bash
python3 -m pytest tests -v
```

The tests need no network access or API keys - AI providers are replaced
with fakes.

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
- Keep the runtime lightweight; do not add heavy dependencies unless they are clearly required, and never ones that depend on PyTorch.
- Preserve the modular design: `nlu/` (matching), `dialogue/` (routing), `ai/` (LLM fallback), `asr/` (speech in), `tts/` (speech out).
- Thresholds and model names belong in `config.py`, not inline.
- Update documentation when behavior or setup changes.
