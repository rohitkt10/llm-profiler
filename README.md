# LLM Inference Profiler

A professional command-line tool that automatically benchmarks and profiles the inference performance of any HuggingFace language model.

## Features

- **Automated Profiling**: Measures memory usage, throughput, latency, and OOM limits.
- **Batch Size Sweep**: Automatically finds the maximum batch size before OOM.
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
  BS=2: ✓ 127.0 tok/s (0.3s)
  BS=4: ✓ 257.3 tok/s (0.3s)
✓ Max successful batch size: 4
```

### Quantization Testing

Test with 4-bit quantization to observe memory savings (0.4 GB vs 0.9 GB) and throughput impact:

```bash
llm-profile --model "Qwen/Qwen2.5-0.5B-Instruct" --quantization 4bit --max-batch-size 4 --max-new-tokens 20
```

**Actual Output:**
```
🔍 Profiling Qwen/Qwen2.5-0.5B-Instruct...
  Configuration: quant=4bit, device=auto, max_bs=4
[1/5] Loading model...
✓ Model loaded: 0.4 GB VRAM
[2/5] Testing batch sizes (up to 4)...
  BS=1: ✓ 45.1 tok/s (0.4s)
  BS=2: ✓ 69.6 tok/s (0.6s)
  BS=4: ✓ 134.8 tok/s (0.6s)
✓ Max successful batch size: 4
```

### Custom Limits

Limit the batch size sweep and generation length:

```bash
llm-profile --model "Qwen/Qwen2.5-0.5B-Instruct" --max-batch-size 16 --max-new-tokens 100
```

## License

MIT