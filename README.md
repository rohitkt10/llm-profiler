# LLM Inference Profiler

A professional command-line tool that automatically benchmarks and profiles the inference performance of any HuggingFace language model.

## Features

- **Automated Profiling**: Measures memory usage, throughput, latency, and OOM limits.
- **Batch Size Sweep**: Automatically finds the maximum batch size before OOM.
- **Prefill vs Decode**: Measures input processing speed vs generation speed.
- **Memory Breakdown**: Analyzes VRAM usage (Weights, KV Cache, Activations).
- **Quantization Support**: Easy testing of 4-bit, 8-bit, fp16, and bf16 precision.
- **Rich Reporting**: Generates detailed terminal output (JSON/HTML reports coming soon).

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
  BS=1: ✓ 67.8 tok/s (0.3s)
  BS=2: ✓ 125.3 tok/s (0.3s)
  BS=4: ✓ 250.8 tok/s (0.3s)
✓ Max successful batch size: 4
[3/5] Measuring prefill vs decode...
  Prefill (100 tokens): 0.02s
  Decode (20 tokens):   0.30s
  Ratio: 18.4x slower
[4/5] Memory profiling...
  Model weights: 0.94 GB
  KV cache (BS=1, 100 tokens): 0.00 GB
  Activation memory: 0.01 GB
  Total: 0.94 GB
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