import base64
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from llm_profiler.profiler import calculate_kv_cache_size

matplotlib.use("Agg")


def get_system_info():
    """Gather system information."""
    info = {
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
        "python_version": os.sys.version.split()[0],
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["total_vram_gb"] = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )
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
            throughputs.append(0)  # Placeholder
            ooms.append(True)

    plt.figure(figsize=(10, 6))

    # Filter successful for line plot
    valid_bs = [bs for i, bs in enumerate(batch_sizes) if not ooms[i]]
    valid_tp = [tp for i, tp in enumerate(throughputs) if not ooms[i]]

    if valid_tp:
        max_tp = max(valid_tp)
        plt.ylim(0, max_tp * 1.2)

    plt.plot(valid_bs, valid_tp, "b-o", label="Throughput")

    # Plot OOMs
    oom_bs = [bs for i, bs in enumerate(batch_sizes) if ooms[i]]
    if oom_bs:
        # Plot OOMs at 0 or roughly where they would be? Just plot at bottom with X
        plt.scatter(
            oom_bs, [0] * len(oom_bs), color="red", marker="x", s=100, label="OOM"
        )

    plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title(f"Throughput vs Batch Size\n{model_name} ({quantization})")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    safe_name = model_name.replace("/", "-")
    filename = f"{safe_name}-{quantization}-throughput.png"
    filepath = plots_dir / filename
    plt.savefig(filepath, dpi=300)
    plt.close()

    return str(filepath)


def plot_memory_breakdown(
    sweep_results,
    output_dir,
    model_name,
    quantization,
    model,
    tokenizer,
    max_new_tokens,
    weights_gb,
):
    """
    Generates stacked bar chart for memory breakdown.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(
        [bs for bs, res in sweep_results.items() if res.get("status") == "success"]
    )
    if not batch_sizes:
        return None

    # Calculate input length roughly
    dummy_text = "This is a test sentence used for profiling."
    input_len = len(tokenizer.encode(dummy_text))
    seq_len = input_len + max_new_tokens  # Total seq len in cache

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

    # Calculate max total memory for ylim scaling
    totals = [w + k + a for w, k, a in zip(weights, kv_caches, activations)]
    max_val = max(totals) if totals else 0

    x = np.arange(len(batch_sizes))
    width = 0.5

    plt.figure(figsize=(10, 6))

    # Set Y-axis limit with buffer
    if max_val > 0:
        plt.ylim(0, max_val * 1.2)

    plt.bar(x, weights, width, label="Weights", color="lightgray")
    plt.bar(x, kv_caches, width, bottom=weights, label="KV Cache", color="skyblue")
    # Bottom for activations is weights + kv_caches
    bottom_act = [w + k for w, k in zip(weights, kv_caches)]
    plt.bar(
        x, activations, width, bottom=bottom_act, label="Activations", color="salmon"
    )

    plt.xlabel("Batch Size")
    plt.ylabel("VRAM Usage (GB)")
    plt.title(f"Memory Breakdown vs Batch Size\n{model_name} ({quantization})")
    plt.xticks(x, batch_sizes)
    plt.grid(True, axis="y", alpha=0.3)
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
            plt.plot(batches, throughputs, marker="o", label=name)

    plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title("Model Comparison: Throughput")
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

    # --- HTML Construction Helper ---
    def dict_to_table(d, title):
        rows = "".join(
            f"<tr><th>{k.replace('_', ' ').title()}</th><td>{v}</td></tr>"
            for k, v in d.items()
        )
        return f"<h3>{title}</h3><table>{rows}</table>"

    # System Info
    sys_info_html = dict_to_table(data.get("system_info", {}), "System Info")

    # Throughput Table
    tp_rows = ""
    throughput_data = data.get("throughput", {})
    # Sort by batch size
    sorted_tp = sorted(
        [
            (int(k.split("_")[1]), v)
            for k, v in throughput_data.items()
            if k.startswith("batch_")
        ],
        key=lambda x: x[0],
    )
    for bs, stats in sorted_tp:
        tp_rows += (
            f"<tr><td>{bs}</td><td>{stats.get('tokens_per_sec', 0):.1f}</td>"
            f"<td>{stats.get('total_time_sec', 0):.2f}</td></tr>"
        )

    throughput_html = f"""
    <h3>Throughput</h3>
    <table>
        <tr><th>Batch Size</th><th>Tokens/sec</th><th>Time (sec)</th></tr>
        {tp_rows}
    </table>
    """

    # Memory Breakdown
    mem = data.get("memory", {})
    mem_rows = ""
    for k, v in mem.items():
        label = k.replace("_gb", "").replace("_", " ").title()
        mem_rows += f"<tr><td>{label}</td><td>{v:.2f} GB</td></tr>"

    memory_html = f"""
    <h3>Memory Breakdown</h3>
    <table>
        <tr><th>Component</th><th>Size</th></tr>
        {mem_rows}
    </table>
    """

    # Prefill/Decode
    pd = data.get("prefill_decode", {})
    pd_html = f"""
    <h3>Prefill vs Decode</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Prefill Time</td><td>{pd.get('prefill_time_sec', 0):.2f} s</td></tr>
        <tr><td>Decode Time</td><td>{pd.get('decode_time_sec', 0):.2f} s</td></tr>
        <tr><td>Ratio</td><td>{pd.get('ratio', 0):.1f}x slower</td></tr>
        <tr><td>Per Token Decode</td>
            <td>{pd.get('per_token_decode_ms', 0):.2f} ms</td></tr>
    </table>
    """

    # Latency Analysis
    lat = data.get("latency", {})

    # Output Length Impact
    out_rows = ""
    for item in lat.get("output_length_impact", []):
        out_rows += (
            f"<tr><td>{item['num_tokens']}</td>"
            f"<td>{item['total_time_sec']:.2f} s</td></tr>"
        )

    latency_html = f"""
    <h3>Output Length Impact</h3>
    <table>
        <tr><th>Tokens</th><th>Total Time</th></tr>
        {out_rows}
    </table>
    """

    # Batch Size Impact (Time per token)
    bs_rows = ""
    for item in lat.get("batch_size_impact", []):
        bs_rows += (
            f"<tr><td>{item['batch_size']}</td>"
            f"<td>{item['time_per_token_ms']:.2f} ms</td></tr>"
        )

    latency_html += f"""
    <h3>Batch Size Impact (Latency)</h3>
    <table>
        <tr><th>Batch Size</th><th>Time/Token</th></tr>
        {bs_rows}
    </table>
    """

    html_content = f"""
    <html>
    <head>
        <title>LLM Profile: {data.get('model_name')}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f9f9f9;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #2980b9;
                margin-top: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            h3 {{
                color: #16a085;
                margin-top: 25px;
                font-size: 1.1em;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin-left: auto;
                margin-right: auto;
            }}
            th, td {{
                border: 1px solid #e0e0e0;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
                color: #555;
                text-transform: uppercase;
                font-size: 0.85em;
            }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
            td {{
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 0.95em;
                color: #444;
            }}
            .plot {{
                text-align: center;
                margin: 30px 0;
                background: white;
                padding: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border-radius: 4px;
            }}
            img {{ max-width: 100%; height: auto; }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                box-shadow: 0 0 15px rgba(0,0,0,0.05);
                border-radius: 8px;
            }}
            .meta-table td {{ font-family: sans-serif; font-weight: 500; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LLM Inference Profile</h1>
            <table class="meta-table">
                <tr><th>Model</th><td>{data.get('model_name')}</td></tr>
                <tr><th>Quantization</th><td>{data.get('quantization')}</td></tr>
                <tr><th>Device</th><td>{data.get('device')}</td></tr>
                <tr><th>Timestamp</th>
                    <td>{data.get('timestamp', datetime.now().isoformat())}</td></tr>
            </table>
            
            <h2>Visualizations</h2>
            <div class="plot">
                <h3>Throughput vs Batch Size</h3>
                <img src="{plot_imgs.get('throughput', '')}" alt="Throughput Plot" />
            </div>
            
            <div class="plot">
                <h3>Memory Breakdown</h3>
                <img src="{plot_imgs.get('memory', '')}" alt="Memory Plot" />
            </div>
            
            <h2>Metrics Summary</h2>
            {sys_info_html}
            {throughput_html}
            {memory_html}
            {pd_html}
            {latency_html}
        </div>
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
