"""
File I/O utilities for disentangle-sycophancy project.
"""
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
    """
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
    
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
    """
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_tensor(filepath: Union[str, Path]) -> torch.Tensor:
    """
    Load a PyTorch tensor from file.
    
    Args:
        filepath: Path to the tensor file
    
    Returns:
        Loaded tensor
    """
    return torch.load(filepath, map_location='cpu')


def save_tensor(tensor: torch.Tensor, filepath: Union[str, Path]) -> None:
    """
    Save a PyTorch tensor to file.
    
    Args:
        tensor: Tensor to save
        filepath: Path to save the file
    """
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(tensor, filepath)


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load a NumPy array from file.
    
    Args:
        filepath: Path to the numpy file (.npy or .npz)
    
    Returns:
        Loaded array
    """
    if filepath.suffix == '.npz':
        return np.load(filepath)
    else:
        return np.load(filepath)


def save_numpy(array: np.ndarray, filepath: Union[str, Path]) -> None:
    """
    Save a NumPy array to file.
    
    Args:
        array: Array to save
        filepath: Path to save the file
    """
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    np.save(filepath, array)


def ensure_dir_exists(dirpath: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dirpath: Path to the directory
    """
    Path(dirpath).mkdir(parents=True, exist_ok=True)


def get_file_stem(filepath: Union[str, Path]) -> str:
    """
    Get the stem (filename without extension) of a file.
    
    Args:
        filepath: Path to the file
    
    Returns:
        File stem
    """
    return Path(filepath).stem


def find_files(
    directory: Union[str, Path], 
    pattern: str = "*", 
    recursive: bool = True
) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        recursive: Whether to search recursively
    
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def load_text_file(filepath: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Load text from a file.
    
    Args:
        filepath: Path to the text file
        encoding: File encoding
    
    Returns:
        File contents as string
    """
    with open(filepath, 'r', encoding=encoding) as f:
        return f.read()


def save_text_file(
    text: str, 
    filepath: Union[str, Path], 
    encoding: str = 'utf-8'
) -> None:
    """
    Save text to a file.
    
    Args:
        text: Text to save
        filepath: Path to save the file
        encoding: File encoding
    """
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding=encoding) as f:
        f.write(text)


def load_jsonl(filepath: Union[str, Path]) -> List[Dict]:
    """
    Load data from a JSONL (JSON Lines) file.
    
    Args:
        filepath: Path to the JSONL file
    
    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: Union[str, Path]) -> None:
    """
    Save data to a JSONL (JSON Lines) file.
    
    Args:
        data: List of dictionaries to save
        filepath: Path to save the file
    """
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_cache_path(
    base_dir: Union[str, Path], 
    *path_components: str
) -> Path:
    """
    Construct a cache file path.
    
    Args:
        base_dir: Base cache directory
        *path_components: Path components to join
    
    Returns:
        Cache file path
    """
    return Path(base_dir) / Path(*path_components)


def file_exists(filepath: Union[str, Path]) -> bool:
    """
    Check if a file exists.
    
    Args:
        filepath: Path to check
    
    Returns:
        True if file exists, False otherwise
    """
    return Path(filepath).exists()


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        filepath: Path to the file
    
    Returns:
        File size in bytes
    """
    return Path(filepath).stat().st_size