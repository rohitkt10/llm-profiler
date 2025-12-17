import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_profiler.utils import generate_batch_sizes

def get_vram_usage() -> float:
    """Returns the current VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def load_model(model_name: str, quantization_config=None, device="auto"):
    """
    Loads a model and tokenizer from HuggingFace.
    
    Args:
        model_name: Name or path of the model.
        quantization_config: BitsAndBytesConfig or torch.dtype (for fp16/bf16).
        device: "cuda", "cpu", or "auto".
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    kwargs = {
        "device_map": "auto" if device == "cuda" else None,
        "trust_remote_code": True,
    }

    # Handle quantization/dtype
    if isinstance(quantization_config, torch.dtype):
        kwargs["torch_dtype"] = quantization_config
    elif quantization_config is not None:
        # It's a BitsAndBytesConfig
        kwargs["quantization_config"] = quantization_config
        # Often with quantization we rely on auto device map or specific setup
    else:
        # Default fallback
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    if device == "cpu" and model.device.type != "cpu":
        model = model.to("cpu")
        
    return model, tokenizer

def measure_throughput(model, tokenizer, batch_size: int = 1, max_new_tokens: int = 50):
    """
    Measures throughput for a given batch size.
    Returns (throughput in tokens/sec, duration in seconds).
    """
    device = model.device
    
    input_text = "This is a test sentence used for profiling."
    inputs = tokenizer([input_text] * batch_size, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        _ = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            min_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    duration = end_time - start_time
    total_tokens = batch_size * max_new_tokens
    throughput = total_tokens / duration
    
    return throughput, duration

def sweep_batch_sizes(model, tokenizer, max_batch_size: int = 128, max_new_tokens: int = 50):
    """
    Sweeps through batch sizes measuring throughput and VRAM.
    Stops upon OOM or 3 consecutive failures (though OOM implies failure).
    """
    batch_sizes = generate_batch_sizes(max_batch_size)
    results = {}
    consecutive_ooms = 0
    
    for bs in batch_sizes:
        if consecutive_ooms >= 3:
            break
            
        try:
            # Clean up before run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            throughput, duration = measure_throughput(model, tokenizer, batch_size=bs, max_new_tokens=max_new_tokens)
            vram = get_vram_usage()
            
            results[bs] = {
                "throughput": throughput,
                "duration": duration,
                "vram_gb": vram,
                "status": "success"
            }
            consecutive_ooms = 0 # Reset on success
            
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            results[bs] = {
                "status": "oom"
            }
            consecutive_ooms += 1
            
        except Exception as e:
            # Handle other errors (like timeout, though timeout is handled in CLI usually)
            results[bs] = {
                "status": "error",
                "error": str(e)
            }
            # Treat other errors as potentially stopping too? 
            # PRD says "consecutive OOMs", but let's be safe.
            # We'll just record error.
            pass
            
    return results

def find_oom_limit(results):
    """
    Finds the maximum successful batch size from sweep results.
    Returns None if no batch size was successful.
    """
    successful_bs = [bs for bs, res in results.items() if res.get("status") == "success"]
    if not successful_bs:
        return None
    return max(successful_bs)