from unittest.mock import MagicMock, patch

import torch

from llm_profiler.profiler import (
    calculate_kv_cache_size,
    find_oom_limit,
    measure_output_length_impact,
    measure_prefill_decode,
    profile_memory_breakdown,
    sweep_batch_sizes,
)


@patch("llm_profiler.profiler.measure_throughput")
@patch("llm_profiler.profiler.get_vram_usage")
@patch("llm_profiler.profiler.generate_batch_sizes")
def test_sweep_batch_sizes_success(mock_gen, mock_vram, mock_throughput):
    """Test a completely successful sweep."""
    # Setup
    mock_gen.return_value = [1, 2, 4]
    mock_throughput.return_value = (10.0, 1.0)  # tok/s, duration
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
        (10.0, 1.0),  # BS 1
        (12.0, 1.0),  # BS 2
        torch.cuda.OutOfMemoryError("OOM"),  # BS 4
        torch.cuda.OutOfMemoryError("OOM"),  # BS 8
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
def test_sweep_batch_sizes_stops_after_consecutive_ooms(
    mock_gen, mock_vram, mock_throughput
):
    """Test that sweep stops after 3 consecutive OOMs."""
    mock_gen.return_value = [1, 2, 4, 8, 16, 32]
    # BS 1 success, then 3 OOMs (2, 4, 8). Should stop and not test 16.
    mock_throughput.side_effect = [
        (10.0, 1.0),  # BS 1
        torch.cuda.OutOfMemoryError("OOM"),  # BS 2
        torch.cuda.OutOfMemoryError("OOM"),  # BS 4
        torch.cuda.OutOfMemoryError("OOM"),  # BS 8
    ]

    results = sweep_batch_sizes(MagicMock(), MagicMock(), max_batch_size=32)

    assert len(results) == 4  # 1, 2, 4, 8
    assert results[8]["status"] == "oom"
    assert 16 not in results  # Should have stopped


def test_find_oom_limit():
    """Test find_oom_limit logic."""
    results = {
        1: {"status": "success"},
        2: {"status": "success"},
        4: {"status": "oom"},
        8: {"status": "oom"},
    }
    assert find_oom_limit(results) == 2


def test_find_oom_limit_all_success():
    results = {1: {"status": "success"}, 2: {"status": "success"}}
    assert find_oom_limit(results) == 2


def test_find_oom_limit_all_oom():
    results = {1: {"status": "oom"}, 2: {"status": "oom"}}
    assert find_oom_limit(results) is None


@patch("time.time")
def test_measure_prefill_decode(mock_time):
    """Test measure_prefill_decode calculation."""
    # Sequence of time.time() calls in measure_prefill_decode:
    # 1. start_prefill
    # 2. end_prefill
    # 3. start_gen
    # 4. end_gen

    # We want:
    # Prefill duration = 0.5s
    # Total Gen duration = 6.0s
    # Decode duration = 6.0 - 0.5 = 5.5s

    mock_time.side_effect = [
        100.0,  # start_prefill
        100.5,  # end_prefill (diff 0.5)
        200.0,  # start_gen
        206.0,  # end_gen (diff 6.0)
    ]

    model = MagicMock()
    model.device = "cpu"
    tokenizer = MagicMock()
    # Mock encoding
    tokenizer.return_value.input_ids = torch.zeros((1, 200))  # Enough tokens

    stats = measure_prefill_decode(model, tokenizer, max_new_tokens=50)

    assert stats["prefill_time_sec"] == 0.5
    assert stats["decode_time_sec"] == 5.5
    assert stats["ratio"] == 11.0  # 5.5 / 0.5
    assert stats["per_token_decode_ms"] == (5.5 / 50) * 1000


# --- Phase 5 Tests ---


def test_calculate_kv_cache_size():
    """Test analytical KV cache calculation."""
    model = MagicMock()
    # Config: 32 layers, 4096 hidden, 32 heads, fp16 (2 bytes)
    model.config.num_hidden_layers = 32
    model.config.hidden_size = 4096
    model.config.num_attention_heads = 32
    # Set explicitly to avoid MagicMock auto-creation
    model.config.num_key_value_heads = 32

    model.dtype = torch.float16

    batch_size = 1
    seq_len = 100

    gb = calculate_kv_cache_size(model, batch_size, seq_len)

    expected_bytes = 2 * 1 * 100 * 32 * 32 * 128 * 2
    expected_gb = expected_bytes / 1024**3

    assert gb == expected_gb


@patch("torch.cuda.max_memory_allocated")
@patch("torch.cuda.memory_allocated")
@patch("torch.cuda.empty_cache")
@patch("torch.cuda.reset_peak_memory_stats")
@patch("llm_profiler.profiler.calculate_kv_cache_size")
def test_profile_memory_breakdown(
    mock_calc, mock_reset, mock_empty, mock_alloc, mock_max_alloc
):
    """Test memory profiling logic."""
    model = MagicMock()
    # Mock device object
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.device = torch.device(device)

    tokenizer = MagicMock()
    tokenizer.return_value.input_ids = torch.zeros((1, 200))

    # Setup mocks
    # weights_mem = 4.0 GB
    # peak_mem = 6.0 GB
    # kv_cache = 0.5 GB
    # activations = 6.0 - 4.0 - 0.5 = 1.5 GB

    mock_alloc.return_value = 4.0 * 1024**3
    mock_max_alloc.return_value = 6.0 * 1024**3
    mock_calc.return_value = 0.5

    stats = profile_memory_breakdown(model, tokenizer)

    assert stats["weights_gb"] == 4.0
    assert stats["total_gb"] == 6.0
    assert stats["kv_cache_gb"] == 0.5
    assert stats["activations_gb"] == 1.5


def test_profile_memory_breakdown_cpu():
    """Test memory profiling returns 0s on CPU."""
    model = MagicMock()
    model.device.type = "cpu"

    stats = profile_memory_breakdown(model, MagicMock())

    assert stats["weights_gb"] == 0.0
    assert stats["total_gb"] == 0.0


@patch("llm_profiler.profiler.measure_throughput")
def test_measure_output_length_impact(mock_throughput):
    """Test latency breakdown by output length."""
    # Lengths: 10, 25, 50, 100, 200
    # Returns (throughput, duration)
    mock_throughput.side_effect = [
        (10.0, 0.1),
        (10.0, 0.25),
        (10.0, 0.5),
        (10.0, 1.0),
        (10.0, 2.0),
    ]

    model = MagicMock()
    tokenizer = MagicMock()

    results = measure_output_length_impact(model, tokenizer)

    assert results[10] == 0.1
    assert results[25] == 0.25
    assert results[50] == 0.5
    assert results[100] == 1.0
    assert results[200] == 2.0
    assert len(results) == 5
