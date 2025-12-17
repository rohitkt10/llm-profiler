import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from llm_profiler.reporter import save_json, get_system_info, plot_throughput, plot_memory_breakdown, save_comparison_json, plot_comparison_throughput, generate_html, generate_markdown

def test_get_system_info():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="Test GPU"), \
         patch("torch.cuda.get_device_properties") as mock_props, \
         patch("psutil.virtual_memory") as mock_mem:
        
        mock_props.return_value.total_memory = 10 * 1024**3
        mock_mem.return_value.total = 16 * 1024**3
        
        info = get_system_info()
        
        assert info["gpu_name"] == "Test GPU"
        assert info["total_vram_gb"] == 10.0
        assert info["system_ram_gb"] == 16.0
        assert "pytorch_version" in info

def test_save_json(tmp_path):
    data = {"model_name": "test/model", "quantization": "none"}
    output_dir = tmp_path
    
    filepath = save_json(data, str(output_dir))
    
    assert os.path.exists(filepath)
    assert "test-model-none" in filepath
    
    with open(filepath, "r") as f:
        loaded = json.load(f)
        assert loaded["model_name"] == "test/model"
        assert "timestamp" in loaded

def test_save_comparison_json(tmp_path):
    data = {"models": ["m1", "m2"]}
    output_dir = tmp_path
    filepath = save_comparison_json(data, str(output_dir))
    
    assert os.path.exists(filepath)
    assert "comparison-" in filepath
    with open(filepath, "r") as f:
        loaded = json.load(f)
        assert loaded["models"] == ["m1", "m2"]

@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_plot_throughput(mock_close, mock_save, tmp_path):
    sweep_results = {
        1: {"status": "success", "throughput": 10.0},
        2: {"status": "success", "throughput": 20.0},
        4: {"status": "oom"}
    }
    
    filepath = plot_throughput(sweep_results, str(tmp_path), "test/model", "none")
    
    assert "test-model-none-throughput.png" in filepath
    mock_save.assert_called_once()
    mock_close.assert_called_once()

@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
@patch("llm_profiler.reporter.calculate_kv_cache_size")
def test_plot_memory_breakdown(mock_calc, mock_close, mock_save, tmp_path):
    sweep_results = {
        1: {"status": "success", "vram_gb": 4.5}, # Weights 4.0 + KV 0.1 + Act 0.4
        2: {"status": "success", "vram_gb": 5.0}, # Weights 4.0 + KV 0.2 + Act 0.8
    }
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1]*10 # length 10
    
    mock_calc.return_value = 0.1 # Simplification: returns constant or we can side_effect
    
    filepath = plot_memory_breakdown(sweep_results, str(tmp_path), "test/model", "none", model, tokenizer, 50, 4.0)
    
    assert "test-model-none-memory.png" in filepath
    mock_save.assert_called_once()
    mock_close.assert_called_once()
    assert mock_calc.call_count == 2 # Called for BS 1 and 2

@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_plot_comparison_throughput(mock_close, mock_save, tmp_path):
    data = {
        "details": [
            {
                "model_name": "m1",
                "throughput": {
                    "batch_1": {"tokens_per_sec": 10.0},
                    "batch_2": {"tokens_per_sec": 20.0}
                }
            },
            {
                "model_name": "m2",
                "throughput": {
                    "batch_1": {"tokens_per_sec": 15.0},
                    "batch_2": {"tokens_per_sec": 25.0}
                }
            }
        ]
    }
    
    filepath = plot_comparison_throughput(data, str(tmp_path))
    assert "comparison-" in filepath
    assert "throughput.png" in filepath
    mock_save.assert_called_once()

def test_generate_html(tmp_path):
    data = {"model_name": "test/model", "quantization": "none", "device": "cuda"}
    output_dir = tmp_path
    
    # Create dummy plot files
    plot_path = tmp_path / "plot.png"
    plot_path.write_bytes(b"fake image data")
    plots = {"throughput": str(plot_path)}
    
    filepath = generate_html(data, str(output_dir), plots)
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".html")
    
    content = Path(filepath).read_text()
    assert "<html>" in content
    assert "test/model" in content
    # Check base64 embedding
    assert "data:image/png;base64," in content

def test_generate_markdown(tmp_path):
    data = {"model_name": "test/model", "quantization": "none", "device": "cuda"}
    output_dir = tmp_path
    
    # Create dummy plot files
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    
    plot_path = plots_dir / "plot.png"
    plot_path.write_bytes(b"fake")
    plots = {"throughput": str(plot_path)}
    
    # generate_markdown saves to profiles/
    # plots are in plots/
    filepath = generate_markdown(data, str(tmp_path), plots)
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".md")
    
    content = Path(filepath).read_text()
    assert "# LLM Inference Profile: test/model" in content
    # Check relative link: ../plots/plot.png
    assert "](../plots/plot.png)" in content or "plots/plot.png" in content