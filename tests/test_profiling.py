import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from llm_profiler.cli import main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_dependencies():
    with patch("llm_profiler.cli.load_model") as mock_load, \
         patch("llm_profiler.cli.sweep_batch_sizes") as mock_sweep, \
         patch("llm_profiler.cli.find_oom_limit") as mock_find_limit, \
         patch("llm_profiler.cli.measure_prefill_decode") as mock_pd, \
         patch("llm_profiler.cli.profile_memory_breakdown") as mock_mem, \
         patch("llm_profiler.cli.measure_output_length_impact") as mock_latency, \
         patch("llm_profiler.cli.get_vram_usage") as mock_vram, \
         patch("llm_profiler.cli.get_system_info") as mock_sys, \
         patch("llm_profiler.cli.save_json") as mock_save, \
         patch("llm_profiler.cli.generate_html") as mock_html, \
         patch("llm_profiler.cli.generate_markdown") as mock_md, \
         patch("llm_profiler.cli.plot_throughput") as mock_plot_tp, \
         patch("llm_profiler.cli.plot_memory_breakdown") as mock_plot_mem, \
         patch("llm_profiler.cli.save_comparison_json") as mock_save_comp, \
         patch("llm_profiler.cli.plot_comparison_throughput") as mock_plot_comp, \
         patch("llm_profiler.cli.check_disk_space") as mock_disk, \
         patch("llm_profiler.cli.manage_cache_size") as mock_cache, \
         patch("llm_profiler.validation.model_info") as mock_info:
        
        # Default successful behaviors
        mock_info.return_value = MagicMock()
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_vram.return_value = 2.5 # GB
        
        # Mock sweep results
        mock_sweep.return_value = {
            1: {"status": "success", "throughput": 15.0, "duration": 3.33, "vram_gb": 2.5}
        }
        mock_find_limit.return_value = 1
        
        # Mock prefill/decode results
        mock_pd.return_value = {
            "prefill_time_sec": 0.5,
            "decode_time_sec": 5.5,
            "ratio": 11.0,
            "per_token_decode_ms": 110.0
        }
        
        # Mock memory breakdown
        mock_mem.return_value = {
            "weights_gb": 4.0,
            "kv_cache_gb": 0.5,
            "activations_gb": 1.5,
            "total_gb": 6.0
        }
        
        # Mock latency
        mock_latency.return_value = {
            10: 0.1, 25: 0.25, 50: 0.5, 100: 1.0, 200: 2.0
        }
        
        # Mock system info
        mock_sys.return_value = {"gpu": "Test"}
        
        # Mock save/plot
        mock_save.return_value = "/path/to/result.json"
        mock_html.return_value = "/path/to/result.html"
        mock_md.return_value = "/path/to/result.md"
        mock_plot_tp.return_value = "/path/to/tp.png"
        mock_plot_mem.return_value = "/path/to/mem.png"
        
        mock_save_comp.return_value = "/path/to/comparison.json"
        mock_plot_comp.return_value = "/path/to/comparison_plot.png"
        
        # Phase 10 defaults
        mock_disk.return_value = (True, 100.0) # Has space
        
        yield {
            "load": mock_load,
            "sweep": mock_sweep,
            "find_limit": mock_find_limit,
            "pd": mock_pd,
            "mem": mock_mem,
            "latency": mock_latency,
            "sys": mock_sys,
            "save": mock_save,
            "html": mock_html,
            "md": mock_md,
            "plot_tp": mock_plot_tp,
            "plot_mem": mock_plot_mem,
            "save_comp": mock_save_comp,
            "plot_comp": mock_plot_comp,
            "disk": mock_disk,
            "cache": mock_cache,
            "vram": mock_vram,
            "info": mock_info
        }

def test_phase10_cpu_fallback(runner, mock_dependencies):
    """Test CPU fallback logic reduces max batch size."""
    with patch("torch.cuda.is_available", return_value=False):
        # We explicitly don't pass --device to test auto detection
        result = runner.invoke(main, ["--model", "Qwen", "--max-batch-size", "128"])
        
        assert result.exit_code == 0
        assert "Running on CPU. Reducing max batch size" in result.output
        assert "max_bs=4" in result.output
        
        # Verify sweep called with 4
        sweep_kwargs = mock_dependencies["sweep"].call_args.kwargs
        assert sweep_kwargs["max_batch_size"] == 4

def test_phase10_disk_space_warning(runner, mock_dependencies):
    """Test warning on low disk space."""
    mock_dependencies["disk"].return_value = (False, 0.05) # Low space
    
    result = runner.invoke(main, ["--model", "Qwen"])
    
    assert result.exit_code == 0
    assert "Warning: Low disk space" in result.output

def test_phase10_cache_management(runner, mock_dependencies):
    """Test cache management called."""
    result = runner.invoke(main, ["--model", "Qwen"])
    assert result.exit_code == 0
    mock_dependencies["cache"].assert_called_once()

def test_phase9_html_output(runner, mock_dependencies):
    """Test HTML output generation."""
    result = runner.invoke(main, ["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--output", "html"])
    
    assert result.exit_code == 0
    assert "HTML report saved to: /path/to/result.html" in result.output
    mock_dependencies["html"].assert_called_once()
    mock_dependencies["save"].assert_not_called()

def test_phase9_markdown_output(runner, mock_dependencies):
    """Test Markdown output generation."""
    result = runner.invoke(main, ["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--output", "markdown"])
    
    assert result.exit_code == 0
    assert "Markdown report saved to: /path/to/result.md" in result.output
    mock_dependencies["md"].assert_called_once()
    mock_dependencies["save"].assert_not_called()

def test_phase7_profiling_output(runner, mock_dependencies):
    """Test full profiling flow including Phase 7 (plotting)."""
    result = runner.invoke(main, ["--model", "Qwen/Qwen2.5-0.5B-Instruct"])
    
    assert result.exit_code == 0
    
    # Check Phase 7 output
    assert "Plot saved to: /path/to/tp.png" in result.output
    assert "Plot saved to: /path/to/mem.png" in result.output
    
    mock_dependencies["plot_tp"].assert_called_once()
    mock_dependencies["plot_mem"].assert_called_once()

def test_phase6_profiling_output(runner, mock_dependencies):
    """Test full profiling flow including Phase 6 (report generation)."""
    result = runner.invoke(main, ["--model", "Qwen/Qwen2.5-0.5B-Instruct"])
    assert result.exit_code == 0
    assert "JSON report saved to: /path/to/result.json" in result.output
    mock_dependencies["save"].assert_called_once()

def test_comparison_mode(runner, mock_dependencies):
    """Test comparison mode (Phase 8)."""
    result = runner.invoke(main, ["--compare", "model1,model2"])
    
    assert result.exit_code == 0
    assert "Starting comparison for 2 models" in result.output
    assert "Comparison JSON saved to" in result.output
    assert "Comparison plot saved to" in result.output
    
    # load_model called twice (once per model)
    assert mock_dependencies["load"].call_count == 2
    mock_dependencies["save_comp"].assert_called_once()
    mock_dependencies["plot_comp"].assert_called_once()

def test_phase3_basic_profiling(runner, mock_dependencies):
    """Test basic profiling command integration with sweep."""
    result = runner.invoke(main, ["--model", "Qwen/Qwen2.5-0.5B-Instruct"])
    assert result.exit_code == 0
    assert "Profiling Qwen/Qwen2.5-0.5B-Instruct" in result.output

def test_phase3_invalid_model(runner, mock_dependencies):
    """Test invalid model handling."""
    from huggingface_hub.utils import RepositoryNotFoundError
    mock_dependencies["info"].side_effect = RepositoryNotFoundError("Not found")
    result = runner.invoke(main, ["--model", "invalid/model"])
    assert result.exit_code != 0
    assert "Model 'invalid/model' not found" in result.output

def test_phase3_quantization_argument(runner, mock_dependencies):
    """Test --quantization argument passed correctly."""
    result = runner.invoke(main, ["--model", "Qwen", "--quantization", "4bit"])
    assert result.exit_code == 0
    assert "quant=4bit" in result.output

def test_phase3_max_new_tokens_argument(runner, mock_dependencies):
    """Test --max-new-tokens argument passed to sweep."""
    result = runner.invoke(main, ["--model", "Qwen", "--max-new-tokens", "20"])
    assert result.exit_code == 0
    sweep_kwargs = mock_dependencies["sweep"].call_args.kwargs
    assert sweep_kwargs["max_new_tokens"] == 20

def test_phase3_oom_reporting(runner, mock_dependencies):
    """Test OOM reporting in CLI output."""
    mock_dependencies["sweep"].return_value = {
        1: {"status": "success", "throughput": 10.0, "duration": 1.0, "vram_gb": 2.0},
        2: {"status": "oom"}
    }
    result = runner.invoke(main, ["--model", "test"])
    assert result.exit_code == 0
    assert "BS=2: ✗ OOM" in result.output.replace("\t", " ")

def test_phase3_no_successful_batches(runner, mock_dependencies):
    """Test output when all batches fail."""
    mock_dependencies["sweep"].return_value = {1: {"status": "oom"}}
    mock_dependencies["find_limit"].return_value = None
    result = runner.invoke(main, ["--model", "test"])
    assert result.exit_code == 0
    assert "No successful batch sizes found" in result.output

def test_invalid_quantization(runner, mock_dependencies):
    result = runner.invoke(main, ["--model", "Qwen", "--quantization", "invalid"])
    assert result.exit_code != 0

def test_invalid_max_batch_size(runner, mock_dependencies):
    result = runner.invoke(main, ["--model", "Qwen", "--max-batch-size", "1000"])
    assert result.exit_code != 0

def test_comparison_mode_placeholder(runner, mock_dependencies):
    # This was renamed to test_comparison_mode but let's keep it if logic differs
    # Actually, duplicates logic. I removed the old one in my mind, but let's ensure no dupe in file.
    # I wrote `test_comparison_mode` above. I'll just check I didn't leave the old one.
    pass

def test_missing_required_args(runner):
    result = runner.invoke(main, [])
    assert result.exit_code != 0