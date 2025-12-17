# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-17

### Added
- **Automated Profiling**: Measure throughput, memory usage, and latency for any HuggingFace model.
- **Batch Size Sweep**: Automatically find the maximum batch size before OOM.
- **Detailed Memory Breakdown**: Visualize usage for Model Weights, KV Cache, and Activations.
- **Prefill vs Decode**: Separate metrics for input processing vs token generation.
- **Reporting**: Generate professional HTML and Markdown reports with embedded plots.
- **Visualization**: Throughput vs Batch Size plots and Memory Breakdown charts.
- **Comparison Mode**: Profile and compare multiple models side-by-side.
- **Robustness**: Automatic CPU fallback, disk space checks, and cache management.
