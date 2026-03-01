import os
from pathlib import Path

def get_actual_case_directory(case_name: str, base_path: str = ".") -> str:
    """
    Auto-detect the actual case directory name based on available exhibit directories.
    
    Args:
        case_name: The case name from the database
        base_path: Base directory to search for case directories
        
    Returns:
        The actual directory name on disk
    """
    # Start with the provided case name
    actual_case_dir = case_name
    
    # Remove common file extensions and suffixes
    case_base = case_name.replace(".ufdr", "").replace(".UFDR", "")
    
    # Look for directories starting with "Exhibit" in the base directory
    try:
        base_path_obj = Path(base_path)
        if not base_path_obj.exists():
            return case_name
            
        # First, try exact match
        for dir_path in base_path_obj.iterdir():
            if dir_path.is_dir():
                dir_name = dir_path.name
                if dir_name == case_name:
                    return dir_name
        
        # Then try various matching patterns
        for dir_path in base_path_obj.iterdir():
            if dir_path.is_dir():
                dir_name = dir_path.name
                
                # Check if it's an Exhibit directory
                if dir_name.startswith("Exhibit"):
                    # Multiple matching strategies
                    patterns_to_check = [
                        case_name in dir_name,
                        dir_name in case_name,
                        case_base in dir_name,
                        dir_name.replace("Exhibit ", "") in case_name,
                        case_name.replace("Exhibit ", "") in dir_name,
                        case_base.replace("Exhibit ", "") in dir_name,
                        dir_name.replace("Exhibit ", "") in case_base
                    ]
                    
                    if any(patterns_to_check):
                        return dir_name
        
        # If no Exhibit directories found, try any directory that might match
        for dir_path in base_path_obj.iterdir():
            if dir_path.is_dir():
                dir_name = dir_path.name
                if case_name in dir_name or dir_name in case_name:
                    return dir_name
                    
    except Exception as e:
        # If directory listing fails, return original case name
        print(f"Error in get_actual_case_directory: {e}")
        pass
    
    return actual_case_dir

def get_case_directory(case_name: str, base_path: str = ".") -> str:
    """
    Alias for get_actual_case_directory for backward compatibility.
    """
    return get_actual_case_directory(case_name, base_path)