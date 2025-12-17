# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phase 1 Implementation** (2025-12-17):
    - Created `pyproject.toml` with project dependencies and metadata.
    - Implemented basic package structure in `llm_profiler/`.
    - Created `cli.py` with minimal `llm-profile` command.
    - Implemented `profiler.py` with model loading and basic throughput measurement.
    - Added `rich` library integration for terminal output.
    - Verified functionality with `Qwen/Qwen2.5-0.5B-Instruct` model.
