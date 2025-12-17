# Contributing to LLM Inference Profiler

Thank you for your interest in contributing to `llm-profiler`! We welcome contributions from the community to help make this tool better for everyone.

## Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rohitkt10/llm-profiler.git
    cd llm-profiler
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies in editable mode:**
    ```bash
    pip install -e .
    pip install pytest
    ```

## Running Tests

We use `pytest` for testing. Ensure all tests pass before submitting a pull request.

```bash
pytest tests/
```

Tests cover:
- CLI argument validation
- Profiling logic (mocked)
- Report generation (HTML/Markdown)
- Edge cases (CPU fallback, disk space)

## Code Style

- Follow PEP 8 guidelines.
- Use meaningful variable names.
- Add type hints where possible.
- Ensure public functions have docstrings.

## Submitting Changes

1.  Fork the repository.
2.  Create a new branch for your feature or fix (`git checkout -b feature/my-new-feature`).
3.  Commit your changes with clear messages (`git commit -m 'feat: add new metric'`).
4.  Push to the branch (`git push origin feature/my-new-feature`).
5.  Open a Pull Request.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. Provide as much detail as possible, including your environment info (OS, Python version, GPU) and the command you ran.
