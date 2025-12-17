# LLM Inference Profiler

A professional command-line tool that automatically benchmarks and profiles the inference performance of any HuggingFace language model.

## Features

- **Automated Profiling**: Measures memory usage, throughput, latency, and OOM limits.
- **Batch Size Sweep**: Automatically finds the maximum batch size before OOM.
- **Prefill vs Decode**: Measures input processing speed vs generation speed.
- **Memory & Latency Breakdown**: Analyzes VRAM usage and generation latency vs output length.
- **Quantization Support**: Easy testing of 4-bit, 8-bit, fp16, and bf16 precision.
- **JSON Reporting**: Generates structured JSON reports with all metrics.

## Installation

```bash
pip install llm-profiler
```

Or for development:

```bash
git clone https://github.com/rohitkt10/llm-profiler.git
cd llm-profiler
pip install -e .
```

## Usage

### Basic Profiling

Profile a model with default settings. Here is a real run on `Qwen/Qwen2.5-0.5B-Instruct`:

```bash
llm-profile --model "Qwen/Qwen2.5-0.5B-Instruct" --max-batch-size 4 --max-new-tokens 20
```

**Actual Output:**
```
🔍 Profiling Qwen/Qwen2.5-0.5B-Instruct...
  Configuration: quant=none, device=auto, max_bs=4
[1/5] Loading model...
✓ Model loaded: 0.9 GB VRAM
[2/5] Testing batch sizes (up to 4)...
  BS=1: ✓ 68.7 tok/s (0.3s)
  BS=2: ✓ 128.9 tok/s (0.3s)
  BS=4: ✓ 247.4 tok/s (0.3s)
✓ Max successful batch size: 4
[3/5] Measuring prefill vs decode...
  Prefill (100 tokens): 0.01s
  Decode (20 tokens):   0.30s
  Ratio: 20.3x slower
[4/5] Memory and latency profiling...
  Model weights: 0.94 GB
  KV cache (BS=1, 100 tokens): 0.00 GB
  Activation memory: 0.01 GB
  Total: 0.94 GB
  Output length impact:
    10 tokens: 0.16s
    25 tokens: 0.39s
    50 tokens: 0.76s
    100 tokens: 1.55s
    200 tokens: 3.20s
[5/5] Generating report...
✓ Results saved to: ~/.llm_profiler/profiles/Qwen-Qwen2.5-0.5B-Instruct-none-20251217-104043.json
```

### Quantization Testing

Test with 4-bit quantization to observe memory savings and throughput impact:

```bash
llm-profile --model "Qwen/Qwen2.5-0.5B-Instruct" --quantization 4bit --max-batch-size 4 --max-new-tokens 20
```

### Custom Limits

Limit the batch size sweep and generation length:

```bash
llm-profile --model "Qwen/Qwen2.5-0.5B-Instruct" --max-batch-size 16 --max-new-tokens 100
```

## License

MIT