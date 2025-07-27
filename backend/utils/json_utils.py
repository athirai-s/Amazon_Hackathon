"""
JSON serialization utilities for EcoAesthetics backend
"""

import numpy as np
from typing import Any, Dict, List, Union

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to JSON-serializable Python types
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python types
    """
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def ensure_json_serializable(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all data in a dictionary is JSON serializable
    
    Args:
        data: Dictionary that may contain numpy types
        
    Returns:
        Dictionary with all numpy types converted
    """
    return convert_numpy_types(data)
