import os

import click
from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError


def validate_model_exists(ctx, param, value):
    """Validates that the model exists on HuggingFace Hub or locally."""
    if not value:
        return value
        
    # Check if local path
    if os.path.isdir(value):
        return value
        
    try:
        model_info(value)
        return value
    except GatedRepoError:
        # We allow it to pass validation here, but it might fail loading later 
        # if token is not set. Loading phase handles the token error more gracefully.
        return value
    except RepositoryNotFoundError:
        raise click.BadParameter(f"Model '{value}' not found on HuggingFace Hub.")
    except Exception as e:
        # Network errors etc.
        raise click.BadParameter(f"Error checking model '{value}': {str(e)}")

def validate_compare_models(ctx, param, value):
    """Validates the comma-separated list of models for comparison."""
    if not value:
        return None
        
    models = [m.strip() for m in value.split(",")]
    if len(models) > 5:
        raise click.BadParameter("Maximum 5 models allowed for comparison.")
        
    for model in models:
        validate_model_exists(ctx, param, model)
        
    return models
