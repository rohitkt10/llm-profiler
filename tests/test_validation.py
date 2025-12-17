import pytest
import click
import torch
from unittest.mock import patch, MagicMock
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
import os
from transformers import BitsAndBytesConfig

from llm_profiler.validation import validate_model_exists, validate_compare_models
from llm_profiler.utils import create_quantization_config, generate_batch_sizes

# --- Tests for validate_model_exists ---

@patch("llm_profiler.validation.model_info")
def test_validate_model_exists_valid_hub_model(mock_model_info):
    """Test with a valid HuggingFace model name."""
    mock_model_info.return_value = MagicMock()
    result = validate_model_exists(None, None, "hf_org/valid_model")
    assert result == "hf_org/valid_model"
    mock_model_info.assert_called_once_with("hf_org/valid_model")

def test_validate_model_exists_valid_local_path(tmp_path):
    """Test with a valid local directory path."""
    local_path = tmp_path / "my_local_model"
    local_path.mkdir()
    result = validate_model_exists(None, None, str(local_path))
    assert result == str(local_path)

@patch("llm_profiler.validation.model_info")
def test_validate_model_exists_invalid_hub_model(mock_model_info):
    """Test with an invalid HuggingFace model name."""
    mock_model_info.side_effect = RepositoryNotFoundError("Model not found")
    with pytest.raises(click.BadParameter, match="Model 'nonexistent/model' not found on HuggingFace Hub."):
        validate_model_exists(None, None, "nonexistent/model")
    mock_model_info.assert_called_once_with("nonexistent/model")

@patch("llm_profiler.validation.model_info")
def test_validate_model_exists_gated_hub_model(mock_model_info):
    """Test with a gated HuggingFace model (should pass validation)."""
    mock_model_info.side_effect = GatedRepoError("Model is gated")
    result = validate_model_exists(None, None, "gated/model")
    assert result == "gated/model"
    mock_model_info.assert_called_once_with("gated/model")

def test_validate_model_exists_empty_value():
    """Test with an empty model name."""
    result = validate_model_exists(None, None, "")
    assert result == ""

@patch("llm_profiler.validation.model_info")
def test_validate_model_exists_generic_error(mock_model_info):
    """Test with a generic exception during model_info call."""
    mock_model_info.side_effect = Exception("Network error")
    with pytest.raises(click.BadParameter, match="Error checking model 'error/model': Network error"):
        validate_model_exists(None, None, "error/model")
    mock_model_info.assert_called_once_with("error/model")

# --- Tests for validate_compare_models ---

@patch("llm_profiler.validation.model_info", return_value=MagicMock())
def test_validate_compare_models_valid_list(mock_model_info):
    """Test with a valid comma-separated list of models."""
    result = validate_compare_models(None, None, "model1,model2,model3")
    assert result == ["model1", "model2", "model3"]
    assert mock_model_info.call_count == 3

@patch("llm_profiler.validation.model_info")
def test_validate_compare_models_too_many_models(mock_model_info):
    """Test with more than 5 models."""
    too_many_models_str = "model1,model2,model3,model4,model5,model6"
    with pytest.raises(click.BadParameter, match="Maximum 5 models allowed for comparison."):
        validate_compare_models(None, None, too_many_models_str)
    mock_model_info.assert_not_called() # Should fail before checking models

@patch("llm_profiler.validation.model_info")
def test_validate_compare_models_with_invalid_model_in_list(mock_model_info):
    """Test with an invalid model within the comma-separated list."""
    mock_model_info.side_effect = [MagicMock(), RepositoryNotFoundError("Invalid model")]
    with pytest.raises(click.BadParameter, match="Model 'invalid/model' not found on HuggingFace Hub."):
        validate_compare_models(None, None, "valid/model,invalid/model")
    assert mock_model_info.call_count == 2 # Called for valid/model then invalid/model

def test_validate_compare_models_empty_value():
    """Test with an empty comparison string."""
    result = validate_compare_models(None, None, "")
    assert result is None

@patch("llm_profiler.validation.model_info")
def test_validate_compare_models_with_whitespace(mock_model_info):
    """Test with models having leading/trailing whitespace."""
    mock_model_info.return_value = MagicMock()
    result = validate_compare_models(None, None, " model1 ,  model2 ")
    assert result == ["model1", "model2"]
    mock_model_info.assert_any_call("model1")
    mock_model_info.assert_any_call("model2")
    assert mock_model_info.call_count == 2

# --- Tests for Quantization Config Creation ---

def test_create_4bit_config():
    """Verify returns BitsAndBytesConfig with load_in_4bit=True."""
    config = create_quantization_config("4bit")
    assert isinstance(config, BitsAndBytesConfig)
    assert config.load_in_4bit is True
    assert config.bnb_4bit_compute_dtype == torch.float16
    assert config.bnb_4bit_quant_type == "nf4"

def test_create_8bit_config():
    """Verify load_in_8bit=True."""
    config = create_quantization_config("8bit")
    assert isinstance(config, BitsAndBytesConfig)
    assert config.load_in_8bit is True

def test_create_fp16_config():
    """Verify returns torch.float16."""
    config = create_quantization_config("fp16")
    assert config == torch.float16

def test_create_none_config():
    """Verify returns None."""
    config = create_quantization_config("none")
    assert config is None

# --- Tests for Batch Size Generation ---

def test_generate_batch_sizes_default():
    """Input max=128, assert [1,2,4,8,16,32,64,128]."""
    result = generate_batch_sizes(128)
    assert result == [1, 2, 4, 8, 16, 32, 64, 128]

def test_generate_batch_sizes_small_max():
    """Input max=16, assert [1,2,4,8,16]."""
    result = generate_batch_sizes(16)
    assert result == [1, 2, 4, 8, 16]

def test_generate_batch_sizes_stops_at_256():
    """Input max=512, assert stops at 256."""
    result = generate_batch_sizes(512)
    assert result[-1] == 256
    assert len(result) == 9 # 1, 2, 4, 8, 16, 32, 64, 128, 256
