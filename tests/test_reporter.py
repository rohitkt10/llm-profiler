import pytest
import json
import os
from unittest.mock import patch, MagicMock
from llm_profiler.reporter import save_json, get_system_info

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
