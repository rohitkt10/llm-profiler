import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_vram_usage() -> float:
    """Returns the current VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def load_model(model_name: str):
    """Loads a model and tokenizer from HuggingFace."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad_token is set as it's required for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model with basic configuration for Phase 1
    # We use device_map="auto" for CUDA to handle large models if possible,
    # though for profiling on one GPU we usually want it on that GPU.
    # float16 is standard for inference on GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True 
    )
    
    if device == "cpu":
        model = model.to("cpu")
        
    return model, tokenizer

def measure_throughput(model, tokenizer, batch_size: int = 1, max_new_tokens: int = 50):
    """
    Measures throughput for a given batch size.
    Returns (throughput in tokens/sec, duration in seconds).
    """
    device = model.device
    
    # Create dummy input. Length doesn't strictly matter for decode throughput 
    # as much as generation length, but we keep it simple.
    input_text = "This is a test sentence used for profiling."
    inputs = tokenizer([input_text] * batch_size, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warmup to settle CUDA kernels
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        # Force generate exactly max_new_tokens for consistent measurement
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
