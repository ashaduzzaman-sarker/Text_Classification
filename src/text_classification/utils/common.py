# ============================================================================
# src/text_classification/utils/common.py
# ============================================================================
import os
import yaml
from pathlib import Path
from typing import Any
from box import ConfigBox
from box.exceptions import BoxValueError
from text_classification.logging.logger import logger


def read_yaml(path: Path) -> ConfigBox:
    """Read YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        ConfigBox with configuration
        
    Raises:
        ValueError: If YAML is empty
        FileNotFoundError: If file doesn't exist
    """
    try:
        with open(path, encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError("YAML file is empty")
            logger.info(f"Loaded config from {path}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"Invalid YAML structure in {path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")


def create_directories(paths: list[Path], verbose: bool = True) -> None:
    """Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
        verbose: Log creation messages
    """
    for path in paths:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


def get_size(path: Path) -> str:
    """Get file size in human-readable format.
    
    Args:
        path: File path
        
    Returns:
        Size string (e.g., "1.5 MB")
    """
    size = os.path.getsize(path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
