import pickle
import os
from typing import Any

def save_to_cache(obj: Any, filename: str, folder: str = "cache") -> None:
    """
    Save any Python object to a cache file using pickle.

    Parameters:
    - obj: Python object to save
    - filename: Filename to save (e.g., 'preprocessing_output.pkl')
    - folder: Folder path (default: 'cache')
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_from_cache(filename: str, folder: str = "cache") -> Any:
    """
    Load a Python object from a cache file.

    Parameters:
    - filename: Filename to load (e.g., 'preprocessing_output.pkl')
    - folder: Folder path (default: 'cache')

    Returns:
    - The loaded Python object
    """
    filepath = os.path.join(folder, filename)
    with open(filepath, "rb") as f:
        return pickle.load(f)

def cache_exists(filename: str, folder: str = "cache") -> bool:
    """
    Check if a cache file exists.

    Parameters:
    - filename: Filename to check
    - folder: Folder path (default: 'cache')

    Returns:
    - True if file exists, else False
    """
    return os.path.exists(os.path.join(folder, filename))

def export_cache_as_bytes(filename: str, folder: str = "cache") -> bytes:
    """
    Load a cache file and return its raw bytes for download.

    This is useful if you want to allow the user to download the .pkl
    file for backup or inspection.

    Returns:
    - Raw bytes of the cached .pkl file.
    """
    filepath = os.path.join(folder, filename)
    with open(filepath, "rb") as f:
        return f.read()
