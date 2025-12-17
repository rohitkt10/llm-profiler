import os
import shutil

import torch
from transformers import BitsAndBytesConfig


def create_quantization_config(quantization: str):
    """
    Creates the appropriate quantization configuration or torch dtype.
    
    Args:
        quantization: One of "4bit", "8bit", "fp16", "none".
        
    Returns:
        BitsAndBytesConfig, torch.dtype, or None.
    """
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif quantization == "fp16":
        return torch.float16
    else:
        return None

def generate_batch_sizes(max_batch_size: int):
    """
    Generates a list of batch sizes (powers of 2) up to max_batch_size.
    Stops at 256 even if max_batch_size is higher, as per PRD logic (implied),
    but user instruction says "stops at 256".
    
    Args:
        max_batch_size: Maximum batch size limit.
        
    Returns:
        List[int]: List of batch sizes.
    """
    batch_sizes = []
    current = 1
    limit = min(max_batch_size, 256)
    
    while current <= limit:
        batch_sizes.append(current)
        current *= 2
        
    return batch_sizes

def check_disk_space(path, min_gb=0.1):
    """
    Checks if available disk space at path is above min_gb.
    Returns (bool, available_gb).
    """
    try:
        # If path doesn't exist, check parent
        p = path
        while p and not os.path.exists(p):
            parent = os.path.dirname(p)
            if parent == p: # Root
                break
            p = parent
        
        if not p or not os.path.exists(p):
            p = "."
            
        total, used, free = shutil.disk_usage(p)
        free_gb = free / 1024**3
        return free_gb >= min_gb, free_gb
    except Exception:
        return True, 100.0 # Fail open

def manage_cache_size(cache_dir, max_files=100):
    """
    Ensures cache directory profiles/ doesn't exceed max_files.
    Deletes oldest files.
    """
    profiles_dir = os.path.join(cache_dir, "profiles")
    if not os.path.exists(profiles_dir):
        return
        
    try:
        files = [os.path.join(profiles_dir, f) for f in os.listdir(profiles_dir) if f.endswith(".json")]
        if len(files) <= max_files:
            return
            
        # Sort by mtime (oldest first)
        files.sort(key=os.path.getmtime)
        
        # Delete oldest
        to_delete = files[:len(files) - max_files]
        for f in to_delete:
            try:
                os.remove(f)
            except OSError:
                pass
    except Exception:
        pass