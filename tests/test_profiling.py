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
         patch("llm_profiler.cli.get_vram_usage") as mock_vram, \
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
        
        yield {
            "load": mock_load,
            "sweep": mock_sweep,
            "find_limit": mock_find_limit,
            "vram": mock_vram,
            "info": mock_info
        }

def test_phase3_basic_profiling(runner, mock_dependencies):
    """Test basic profiling command integration with sweep."""
    result = runner.invoke(main, ["--model", "Qwen/Qwen2.5-0.5B-Instruct"])
    
    assert result.exit_code == 0
    assert "Profiling Qwen/Qwen2.5-0.5B-Instruct" in result.output
    assert "Model loaded: 2.5 GB VRAM" in result.output
    # Check for Phase 3 specific output
    assert "Testing batch sizes" in result.output
    # Match rich output (seems to use space instead of tab in test capture)
    assert "BS=1: ✓ 15.0 tok/s" in result.output.replace("\t", " ") 
    assert "Max successful batch size: 1" in result.output
    
    mock_dependencies["load"].assert_called_once()
    mock_dependencies["sweep"].assert_called_once()

# ... (skip to next replacement)

def test_phase3_oom_reporting(runner, mock_dependencies):
    """Test OOM reporting in CLI output."""
    mock_dependencies["sweep"].return_value = {
        1: {"status": "success", "throughput": 10.0, "duration": 1.0, "vram_gb": 2.0},
        2: {"status": "oom"}
    }
    mock_dependencies["find_limit"].return_value = 1
    
    result = runner.invoke(main, ["--model", "test/model"])
    
    assert result.exit_code == 0
    assert "BS=1: ✓ 10.0 tok/s" in result.output.replace("\t", " ")
    assert "BS=2: ✗ OOM" in result.output.replace("\t", " ")
    assert "Max successful batch size: 1" in result.output

def test_phase3_no_successful_batches(runner, mock_dependencies):
    """Test output when all batches fail."""
    mock_dependencies["sweep"].return_value = {
        1: {"status": "oom"}
    }
    mock_dependencies["find_limit"].return_value = None
    
    result = runner.invoke(main, ["--model", "test/model"])
    
    assert result.exit_code == 0
    assert "BS=1: ✗ OOM" in result.output.replace("\t", " ")
    assert "No successful batch sizes found" in result.output

def test_invalid_quantization(runner, mock_dependencies): # Patched model_info
    """Test invalid quantization option."""
    result = runner.invoke(main, [
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--quantization", "invalid_bit"
    ])
    
    assert result.exit_code != 0
    assert "Invalid value for '--quantization'" in result.output

def test_invalid_max_batch_size(runner, mock_dependencies): # Patched model_info
    """Test invalid max-batch-size (out of range)."""
    result = runner.invoke(main, [
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--max-batch-size", "1000"
    ])
    
    assert result.exit_code != 0
    assert "Invalid value for '--max-batch-size'" in result.output

def test_comparison_mode_placeholder(runner, mock_dependencies):
    """Test comparison mode (Phase 8 placeholder)."""
    result = runner.invoke(main, [
        "--compare", "model1,model2"
    ])
    
    assert result.exit_code == 0
    assert "Starting comparison for models: model1, model2" in result.output
    
    mock_dependencies["load"].assert_not_called()

def test_missing_required_args(runner):
    """Test failure when neither --model nor --compare is provided."""
    result = runner.invoke(main, [])
    assert result.exit_code != 0
    assert "Either --model or --compare must be provided" in result.output