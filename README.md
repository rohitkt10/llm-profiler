# LLM Inference Profiler

A professional command-line tool that automatically benchmarks and profiles the inference performance of any HuggingFace language model.

## Features

- **Automated Profiling**: Measures memory usage, throughput, latency, and OOM limits.
- **Batch Size Sweep**: Automatically finds the maximum batch size before OOM.
- **Quantization Support**: easy testing of 4-bit, 8-bit, fp16, and bf16 precision.
- **Rich Reporting**: Generates JSON, HTML, and Markdown reports with visualizations.

## Installation

```bash
pip install llm-profiler
```

Or for development:

```bash
git clone https://github.com/rohitpc/llm-profiler.git
cd llm-profiler
pip install -e .
```

## Usage

### Basic Profiling

Profile a model with default settings:

```bash
llm-profile --model "Qwen/Qwen2.5-0.5B-Instruct"
```

### Demo Output

```
🔍 Profiling Qwen/Qwen2.5-0.5B-Instruct...
[1/2] Loading model...
✓ Model loaded: 1.2 GB VRAM
[2/2] Measuring throughput...
✓ Batch size 1: 66.4 tok/s (0.8s for 50 tokens)
```

## License

MIT
