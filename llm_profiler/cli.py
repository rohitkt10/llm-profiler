import click
from rich.console import Console
from llm_profiler.profiler import load_model, measure_throughput, get_vram_usage

console = Console()

@click.command()
@click.option("--model", required=True, help="HuggingFace model name or local path.")
def main(model):
    """LLM Inference Profiler."""
    console.print(f"🔍 Profiling {model}...")

    # Load model
    console.print("[1/2] Loading model...", style="bold blue")
    try:
        model_obj, tokenizer = load_model(model)
        vram = get_vram_usage()
        console.print(f"✓ Model loaded: {vram:.1f} GB VRAM", style="green")
    except Exception as e:
        console.print(f"❌ Error loading model: {e}", style="red")
        return

    # Measure throughput
    console.print("[2/2] Measuring throughput...", style="bold blue")
    try:
        throughput, duration = measure_throughput(model_obj, tokenizer, batch_size=1)
        console.print(f"✓ Batch size 1: {throughput:.1f} tok/s ({duration:.1f}s for 50 tokens)", style="green")
    except Exception as e:
        console.print(f"❌ Error measuring throughput: {e}", style="red")

if __name__ == "__main__":
    main()
