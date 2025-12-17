import pytest
import torch
from unittest.mock import MagicMock, patch
from llm_profiler.profiler import sweep_batch_sizes, find_oom_limit

@patch("llm_profiler.profiler.measure_throughput")
@patch("llm_profiler.profiler.get_vram_usage")
@patch("llm_profiler.profiler.generate_batch_sizes")
def test_sweep_batch_sizes_success(mock_gen, mock_vram, mock_throughput):
    """Test a completely successful sweep."""
    # Setup
    mock_gen.return_value = [1, 2, 4]
    mock_throughput.return_value = (10.0, 1.0) # tok/s, duration
    mock_vram.return_value = 2.0
    
    model = MagicMock()
    tokenizer = MagicMock()
    
    results = sweep_batch_sizes(model, tokenizer, max_batch_size=4)
    
    assert len(results) == 3
    assert results[1]["status"] == "success"
    assert results[2]["status"] == "success"
    assert results[4]["status"] == "success"
    assert results[1]["throughput"] == 10.0

@patch("llm_profiler.profiler.measure_throughput")
@patch("llm_profiler.profiler.get_vram_usage")
@patch("llm_profiler.profiler.generate_batch_sizes")
def test_sweep_batch_sizes_with_oom(mock_gen, mock_vram, mock_throughput):
    """Test sweep handling OOM."""
    mock_gen.return_value = [1, 2, 4, 8]
    # Success for 1, 2; OOM for 4, 8
    mock_throughput.side_effect = [
        (10.0, 1.0), # BS 1
        (12.0, 1.0), # BS 2
        torch.cuda.OutOfMemoryError("OOM"), # BS 4
        torch.cuda.OutOfMemoryError("OOM")  # BS 8
    ]
    mock_vram.return_value = 2.0
    
    results = sweep_batch_sizes(MagicMock(), MagicMock(), max_batch_size=8)
    
    assert results[1]["status"] == "success"
    assert results[2]["status"] == "success"
    assert results[4]["status"] == "oom"
    assert results[8]["status"] == "oom"

@patch("llm_profiler.profiler.measure_throughput")
@patch("llm_profiler.profiler.get_vram_usage")
@patch("llm_profiler.profiler.generate_batch_sizes")
def test_sweep_batch_sizes_stops_after_consecutive_ooms(mock_gen, mock_vram, mock_throughput):
    """Test that sweep stops after 3 consecutive OOMs."""
    mock_gen.return_value = [1, 2, 4, 8, 16, 32]
    # BS 1 success, then 3 OOMs (2, 4, 8). Should stop and not test 16.
    mock_throughput.side_effect = [
        (10.0, 1.0), # BS 1
        torch.cuda.OutOfMemoryError("OOM"), # BS 2
        torch.cuda.OutOfMemoryError("OOM"), # BS 4
        torch.cuda.OutOfMemoryError("OOM")  # BS 8
    ]
    
    results = sweep_batch_sizes(MagicMock(), MagicMock(), max_batch_size=32)
    
    assert len(results) == 4 # 1, 2, 4, 8
    assert results[8]["status"] == "oom"
    assert 16 not in results # Should have stopped

def test_find_oom_limit():
    """Test find_oom_limit logic."""
    results = {
        1: {"status": "success"},
        2: {"status": "success"},
        4: {"status": "oom"},
        8: {"status": "oom"}
    }
    assert find_oom_limit(results) == 2

def test_find_oom_limit_all_success():
    results = {1: {"status": "success"}, 2: {"status": "success"}}
    assert find_oom_limit(results) == 2

def test_find_oom_limit_all_oom():
    results = {1: {"status": "oom"}, 2: {"status": "oom"}}
    assert find_oom_limit(results) is None
