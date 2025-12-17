import json
import os
import torch
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import base64
from datetime import datetime
from pathlib import Path
from llm_profiler.profiler import calculate_kv_cache_size

def get_system_info():
    """Gather system information."""
    info = {
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
        "python_version": os.sys.version.split()[0]
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["total_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        info["gpu_name"] = "N/A"
        info["total_vram_gb"] = 0.0
        
    # RAM
    info["system_ram_gb"] = psutil.virtual_memory().total / 1024**3
    
    return info

def save_json(data, output_dir):
    """
    Saves profiling results to a JSON file.
    
    Args:
        data: Dictionary containing all profiling results.
        output_dir: Base directory for output (e.g. ~/.llm_profiler/).
    """
    profiles_dir = Path(output_dir) / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = data.get("model_name", "unknown").replace("/", "-")
    quant = data.get("quantization", "none")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    filename = f"{model_name}-{quant}-{timestamp}.json"
    filepath = profiles_dir / filename
    
    # Add timestamp to data if not present (using ISO format for data content)
    if "timestamp" not in data:
        data["timestamp"] = datetime.now().isoformat()
        
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
        
    return str(filepath)

def plot_throughput(sweep_results, output_dir, model_name, quantization):
    """
    Generates throughput vs batch size plot.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    batch_sizes = sorted([bs for bs in sweep_results.keys()])
    throughputs = []
    ooms = []
    
    for bs in batch_sizes:
        res = sweep_results[bs]
        if res.get("status") == "success":
            throughputs.append(res["throughput"])
            ooms.append(False)
        else:
            throughputs.append(0) # Placeholder
            ooms.append(True)
            
    plt.figure(figsize=(10, 6))
    
    # Filter successful for line plot
    valid_bs = [bs for i, bs in enumerate(batch_sizes) if not ooms[i]]
    valid_tp = [tp for i, tp in enumerate(throughputs) if not ooms[i]]
    
    plt.plot(valid_bs, valid_tp, 'b-o', label='Throughput')
    
    # Plot OOMs
    oom_bs = [bs for i, bs in enumerate(batch_sizes) if ooms[i]]
    if oom_bs:
        # Plot OOMs at 0 or roughly where they would be? Just plot at bottom with X
        plt.scatter(oom_bs, [0]*len(oom_bs), color='red', marker='x', s=100, label='OOM')
        
    plt.xscale('log', base=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title(f'Throughput vs Batch Size\n{model_name} ({quantization})')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    safe_name = model_name.replace("/", "-")
    filename = f"{safe_name}-{quantization}-throughput.png"
    filepath = plots_dir / filename
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return str(filepath)

def plot_memory_breakdown(sweep_results, output_dir, model_name, quantization, model, tokenizer, max_new_tokens, weights_gb):
    """
    Generates stacked bar chart for memory breakdown.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    batch_sizes = sorted([bs for bs, res in sweep_results.items() if res.get("status") == "success"])
    if not batch_sizes:
        return None
        
    # Calculate input length roughly
    dummy_text = "This is a test sentence used for profiling."
    input_len = len(tokenizer.encode(dummy_text))
    seq_len = input_len + max_new_tokens # Total seq len in cache
    
    weights = []
    kv_caches = []
    activations = []
    
    for bs in batch_sizes:
        total = sweep_results[bs]["vram_gb"]
        
        # Calculate components
        kv = calculate_kv_cache_size(model, bs, seq_len)
        act = max(0, total - weights_gb - kv)
        
        weights.append(weights_gb)
        kv_caches.append(kv)
        activations.append(act)
        
    x = np.arange(len(batch_sizes))
    width = 0.5
    
    plt.figure(figsize=(10, 6))
    
    p1 = plt.bar(x, weights, width, label='Weights', color='lightgray')
    p2 = plt.bar(x, kv_caches, width, bottom=weights, label='KV Cache', color='skyblue')
    # Bottom for activations is weights + kv_caches
    bottom_act = [w + k for w, k in zip(weights, kv_caches)]
    p3 = plt.bar(x, activations, width, bottom=bottom_act, label='Activations', color='salmon')
    
    plt.xlabel('Batch Size')
    plt.ylabel('VRAM Usage (GB)')
    plt.title(f'Memory Breakdown vs Batch Size\n{model_name} ({quantization})')
    plt.xticks(x, batch_sizes)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    
    safe_name = model_name.replace("/", "-")
    filename = f"{safe_name}-{quantization}-memory.png"
    filepath = plots_dir / filename
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return str(filepath)

def save_comparison_json(comparison_data, output_dir):
    """Saves comparison results to JSON."""
    profiles_dir = Path(output_dir) / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"comparison-{timestamp}.json"
    filepath = profiles_dir / filename
    
    if "timestamp" not in comparison_data:
        comparison_data["timestamp"] = datetime.now().isoformat()
    
    with open(filepath, "w") as f:
        json.dump(comparison_data, f, indent=2)
        
    return str(filepath)

def plot_comparison_throughput(comparison_data, output_dir):
    """Generates a comparison plot for throughput vs batch size."""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"comparison-{timestamp}-throughput.png"
    filepath = plots_dir / filename
    
    plt.figure(figsize=(12, 8))
    
    if "details" not in comparison_data:
        return None
        
    for model_data in comparison_data["details"]:
        name = model_data["model_name"]
        throughput_data = model_data.get("throughput", {})
        
        batches = []
        throughputs = []
        for key, val in throughput_data.items():
            if key.startswith("batch_"):
                bs = int(key.split("_")[1])
                tp = val["tokens_per_sec"]
                batches.append(bs)
                throughputs.append(tp)
        
        if batches:
            zipped = sorted(zip(batches, throughputs))
            batches, throughputs = zip(*zipped)
            plt.plot(batches, throughputs, marker='o', label=name)
            
    plt.xscale('log', base=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Model Comparison: Throughput')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return str(filepath)

def generate_html(data, output_dir, plots):
    """Generates an HTML report."""
    profiles_dir = Path(output_dir) / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = data.get("model_name", "unknown").replace("/", "-")
    quant = data.get("quantization", "none")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    filename = f"{model_name}-{quant}-{timestamp}.html"
    filepath = profiles_dir / filename
    
    plot_imgs = {}
    for key, path in plots.items():
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                img_data = f.read()
                b64 = base64.b64encode(img_data).decode("utf-8")
                plot_imgs[key] = f"data:image/png;base64,{b64}"
    
    html_content = f"""
    <html>
    <head>
        <title>LLM Profile: {data.get('model_name')}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>LLM Inference Profile</h1>
        <table>
            <tr><th>Model</th><td>{data.get('model_name')}</td></tr>
            <tr><th>Quantization</th><td>{data.get('quantization')}</td></tr>
            <tr><th>Device</th><td>{data.get('device')}</td></tr>
            <tr><th>Timestamp</th><td>{data.get('timestamp', datetime.now().isoformat())}</td></tr>
        </table>
        
        <h2>Throughput</h2>
        <div class="plot">
            <img src="{plot_imgs.get('throughput', '')}" alt="Throughput Plot" />
        </div>
        
        <h2>Memory Breakdown</h2>
        <div class="plot">
            <img src="{plot_imgs.get('memory', '')}" alt="Memory Plot" />
        </div>
        
        <h2>Metrics Summary</h2>
        <pre>{json.dumps(data, indent=2)}</pre>
    </body>
    </html>
    """
    
    with open(filepath, "w") as f:
        f.write(html_content)
        
    return str(filepath)

def generate_markdown(data, output_dir, plots):
    """Generates a Markdown report."""
    profiles_dir = Path(output_dir) / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = data.get("model_name", "unknown").replace("/", "-")
    quant = data.get("quantization", "none")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    filename = f"{model_name}-{quant}-{timestamp}.md"
    filepath = profiles_dir / filename
    
    plot_links = {}
    for key, path in plots.items():
        if path:
            try:
                rel_path = os.path.relpath(path, profiles_dir)
                plot_links[key] = rel_path
            except ValueError:
                plot_links[key] = path
    
    md_content = f"""
# LLM Inference Profile: {data.get('model_name')}

- **Quantization:** {data.get('quantization')}
- **Device:** {data.get('device')}
- **Timestamp:** {data.get('timestamp', datetime.now().isoformat())}

## Throughput

![Throughput Plot]({plot_links.get('throughput', '')})

## Memory Breakdown

![Memory Plot]({plot_links.get('memory', '')})

## Detailed Metrics

```json
{json.dumps(data, indent=2)}
```
"""
    
    with open(filepath, "w") as f:
        f.write(md_content)
        
    return str(filepath)