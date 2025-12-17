import click
import os
import torch
import gc
from pathlib import Path
from rich.console import Console
from llm_profiler.profiler import load_model, measure_throughput, get_vram_usage, sweep_batch_sizes, find_oom_limit, measure_prefill_decode, profile_memory_breakdown, measure_output_length_impact
from llm_profiler.validation import validate_model_exists, validate_compare_models
from llm_profiler.utils import create_quantization_config
from llm_profiler.reporter import get_system_info, save_json, plot_throughput, plot_memory_breakdown, save_comparison_json, plot_comparison_throughput

console = Console()

def profile_single_model(model, quantization, max_batch_size, max_new_tokens, device, output, cache_dir):
    """Profiles a single model and returns the results dictionary."""
    console.print(f"🔍 Profiling {model}...")
    console.print(f"  Configuration: quant={quantization}, device={device}, max_bs={max_batch_size}")

    profiling_data = {
        "model_name": model,
        "quantization": quantization,
        "device": device,
        "system_info": get_system_info()
    }

    # 1. Load model
    console.print("[1/5] Loading model...", style="bold blue")
    model_obj = None
    tokenizer = None
    weights_gb = 0.0
    try:
        quant_config = create_quantization_config(quantization)
        model_obj, tokenizer = load_model(model, quantization_config=quant_config, device=device)
        vram = get_vram_usage()
        console.print(f"✓ Model loaded: {vram:.1f} GB VRAM", style="green")
    except Exception as e:
        console.print(f"❌ Error loading model: {e}", style="red")
        return None

    # 2. Sweep batch sizes
    console.print(f"[2/5] Testing batch sizes (up to {max_batch_size})...", style="bold blue")
    results = {}
    try:
        results = sweep_batch_sizes(model_obj, tokenizer, max_batch_size=max_batch_size, max_new_tokens=max_new_tokens)
        
        throughput_data = {}
        for bs, res in results.items():
            if res.get("status") == "success":
                console.print(f"  BS={bs}:\t✓ {res['throughput']:.1f} tok/s ({res['duration']:.1f}s)", style="green")
                throughput_data[f"batch_{bs}"] = {
                    "tokens_per_sec": res['throughput'],
                    "total_time_sec": res['duration']
                }
            elif res.get("status") == "oom":
                console.print(f"  BS={bs}:\t✗ OOM", style="red")
            else:
                console.print(f"  BS={bs}:\t✗ Error: {res.get('error')}", style="red")
        
        profiling_data["throughput"] = throughput_data
        
        oom_limit = find_oom_limit(results)
        profiling_data["oom_limit"] = oom_limit
        if oom_limit:
            console.print(f"✓ Max successful batch size: {oom_limit}", style="bold green")
        else:
            console.print("❌ No successful batch sizes found.", style="bold red")

    except Exception as e:
        console.print(f"❌ Error during batch sweep: {e}", style="red")

    # 3. Prefill vs Decode
    console.print("[3/5] Measuring prefill vs decode...", style="bold blue")
    try:
        pd_stats = measure_prefill_decode(model_obj, tokenizer, max_new_tokens=max_new_tokens)
        console.print(f"  Prefill (100 tokens): {pd_stats['prefill_time_sec']:.2f}s")
        console.print(f"  Decode ({max_new_tokens} tokens):   {pd_stats['decode_time_sec']:.2f}s")
        console.print(f"  Ratio: {pd_stats['ratio']:.1f}x slower")
        profiling_data["prefill_decode"] = pd_stats
    except Exception as e:
        console.print(f"⚠️  Warning: Failed to measure prefill/decode: {e}", style="yellow")

    # 4. Memory and Latency Breakdown
    console.print("[4/5] Memory and latency profiling...", style="bold blue")
    try:
        mem_stats = profile_memory_breakdown(model_obj, tokenizer, batch_size=1, seq_len=100)
        weights_gb = mem_stats['weights_gb']
        console.print(f"  Model weights: {mem_stats['weights_gb']:.2f} GB")
        console.print(f"  KV cache (BS=1, 100 tokens): {mem_stats['kv_cache_gb']:.2f} GB")
        console.print(f"  Activation memory: {mem_stats['activations_gb']:.2f} GB")
        console.print(f"  Total: {mem_stats['total_gb']:.2f} GB")
        profiling_data["memory"] = mem_stats
        
        console.print("  Output length impact:")
        latency_stats = measure_output_length_impact(model_obj, tokenizer)
        
        output_impact_list = []
        for length, duration in latency_stats.items():
            if duration is not None:
                console.print(f"    {length} tokens: {duration:.2f}s")
                output_impact_list.append({"num_tokens": length, "total_time_sec": duration})
            else:
                console.print(f"    {length} tokens: Failed", style="red")
        
        batch_impact_list = []
        for bs, res in results.items():
            if res.get("status") == "success":
                total_tokens = bs * max_new_tokens
                time_per_token_ms = (res['duration'] / total_tokens) * 1000
                batch_impact_list.append({"batch_size": bs, "time_per_token_ms": time_per_token_ms})
        
        profiling_data["latency"] = {
            "batch_size_impact": batch_impact_list,
            "output_length_impact": output_impact_list
        }
        
    except Exception as e:
        console.print(f"⚠️  Warning: Failed to measure memory/latency: {e}", style="yellow")

    # 5. Report Generation
    console.print("[5/5] Generating report...", style="bold blue")
    try:
        saved_json = save_json(profiling_data, cache_dir)
        console.print(f"✓ Results saved to: {saved_json}", style="bold green")
        
        plot_tp = plot_throughput(results, cache_dir, model, quantization)
        console.print(f"✓ Plot saved to: {plot_tp}", style="bold green")
        
        plot_mem = plot_memory_breakdown(results, cache_dir, model, quantization, model_obj, tokenizer, max_new_tokens, weights_gb)
        if plot_mem:
            console.print(f"✓ Plot saved to: {plot_mem}", style="bold green")
            
    except Exception as e:
        console.print(f"❌ Error generating report: {e}", style="red")

    # Cleanup
    if model_obj is not None:
        del model_obj
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return profiling_data

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
        console.print(f"🔍 Starting comparison for {len(compare)} models: {', '.join(compare)}")
        
        all_results = []
        for model_name in compare:
            console.print(f"\n--- Profiling {model_name} ---", style="bold")
            res = profile_single_model(model_name, quantization, max_batch_size, max_new_tokens, device, output, cache_dir)
            if res:
                all_results.append(res)
            
            # Additional cleanup between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        if all_results:
            console.print("\n🔍 Generating Comparison Report...", style="bold blue")
            
            # Construct comparison data structure
            comparison_data = {
                "models": compare,
                "quantization": quantization,
                "details": all_results,
                "comparison": {
                    "memory": {},
                    "throughput_bs1": {},
                    "oom_limit": {}
                }
            }
            
            # Summarize metrics
            for res in all_results:
                m_name = res["model_name"]
                if "memory" in res:
                    comparison_data["comparison"]["memory"][m_name] = res["memory"].get("total_gb", 0)
                if "throughput" in res and "batch_1" in res["throughput"]:
                    comparison_data["comparison"]["throughput_bs1"][m_name] = res["throughput"]["batch_1"].get("tokens_per_sec", 0)
                comparison_data["comparison"]["oom_limit"][m_name] = res.get("oom_limit")

            try:
                saved_path = save_comparison_json(comparison_data, cache_dir)
                console.print(f"✓ Comparison JSON saved to: {saved_path}", style="bold green")
                
                plot_path = plot_comparison_throughput(comparison_data, cache_dir)
                if plot_path:
                    console.print(f"✓ Comparison plot saved to: {plot_path}", style="bold green")
            except Exception as e:
                console.print(f"❌ Error saving comparison report: {e}", style="red")
        
        return

    # Single model mode
    profile_single_model(model, quantization, max_batch_size, max_new_tokens, device, output, cache_dir)

if __name__ == "__main__":
    main()
