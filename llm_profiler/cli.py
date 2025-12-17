import click
import os
import torch
from pathlib import Path
from rich.console import Console
from llm_profiler.profiler import load_model, measure_throughput, get_vram_usage, sweep_batch_sizes, find_oom_limit, measure_prefill_decode
from llm_profiler.validation import validate_model_exists, validate_compare_models
from llm_profiler.utils import create_quantization_config

console = Console()

@click.command()
@click.option("--model", required=False, callback=validate_model_exists, help="HuggingFace model name or local path.")
@click.option("--quantization", type=click.Choice(["4bit", "8bit", "fp16", "none"]), default="none", help="Quantization level.")
@click.option("--max-batch-size", type=click.IntRange(1, 512), default=128, help="Maximum batch size to test.")
@click.option("--max-new-tokens", type=click.IntRange(1, 1000), default=50, help="Number of tokens to generate per benchmark.")
@click.option("--device", type=click.Choice(["cuda", "cpu", "auto"]), default="auto", help="Device to run inference on.")
@click.option("--output", type=click.Choice(["json", "html", "markdown"]), default="json", help="Output format for results.")
@click.option("--cache-dir", type=click.Path(file_okay=False, dir_okay=True, writable=True), default=os.path.expanduser("~/.llm_profiler/"), help="Directory for saving results.")
@click.option("--compare", callback=validate_compare_models, help="Comma-separated list of up to 5 models for comparison.")
@click.option("--resume", is_flag=True, help="Resume interrupted profiling session.")
@click.option("--timeout", type=click.IntRange(60, 3600), default=300, help="Timeout in seconds per model.")
def main(model, quantization, max_batch_size, max_new_tokens, device, output, cache_dir, compare, resume, timeout):
    """LLM Inference Profiler."""
    
    if not model and not compare:
        ctx = click.get_current_context()
        ctx.fail("Either --model or --compare must be provided.")

    os.makedirs(cache_dir, exist_ok=True)

    if compare:
        console.print(f"🔍 Starting comparison for models: {', '.join(compare)}")
        console.print("⚠️  Comparison mode logic to be implemented in Phase 8.", style="yellow")
        return

    console.print(f"🔍 Profiling {model}...")
    console.print(f"  Configuration: quant={quantization}, device={device}, max_bs={max_batch_size}")

    # Load model
    console.print("[1/5] Loading model...", style="bold blue")
    try:
        quant_config = create_quantization_config(quantization)
        model_obj, tokenizer = load_model(model, quantization_config=quant_config, device=device)
        vram = get_vram_usage()
        console.print(f"✓ Model loaded: {vram:.1f} GB VRAM", style="green")
    except Exception as e:
        console.print(f"❌ Error loading model: {e}", style="red")
        return

    # Sweep batch sizes (Phase 3)
    console.print(f"[2/5] Testing batch sizes (up to {max_batch_size})...", style="bold blue")
    try:
        results = sweep_batch_sizes(model_obj, tokenizer, max_batch_size=max_batch_size, max_new_tokens=max_new_tokens)
        
        for bs, res in results.items():
            if res.get("status") == "success":
                console.print(f"  BS={bs}:\t✓ {res['throughput']:.1f} tok/s ({res['duration']:.1f}s)", style="green")
            elif res.get("status") == "oom":
                console.print(f"  BS={bs}:\t✗ OOM", style="red")
            else:
                console.print(f"  BS={bs}:\t✗ Error: {res.get('error')}", style="red")
        
        oom_limit = find_oom_limit(results)
        if oom_limit:
            console.print(f"✓ Max successful batch size: {oom_limit}", style="bold green")
        else:
            console.print("❌ No successful batch sizes found.", style="bold red")

    except Exception as e:
        console.print(f"❌ Error during batch sweep: {e}", style="red")

    # Prefill vs Decode (Phase 4)
    console.print("[3/5] Measuring prefill vs decode...", style="bold blue")
    try:
        pd_stats = measure_prefill_decode(model_obj, tokenizer, max_new_tokens=max_new_tokens)
        console.print(f"  Prefill (100 tokens): {pd_stats['prefill_time_sec']:.2f}s")
        console.print(f"  Decode ({max_new_tokens} tokens):   {pd_stats['decode_time_sec']:.2f}s")
        console.print(f"  Ratio: {pd_stats['ratio']:.1f}x slower")
    except Exception as e:
        console.print(f"⚠️  Warning: Failed to measure prefill/decode: {e}", style="yellow")

if __name__ == "__main__":
    main()