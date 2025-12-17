import json
import os
import torch
import psutil
from datetime import datetime
from pathlib import Path

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
