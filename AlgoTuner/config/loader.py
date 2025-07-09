import os
from pathlib import Path
import threading # Import threading for lock

# Handle optional YAML dependency
try:
    import yaml
except ImportError:
    yaml = None

_DEFAULT_CONFIG_FILENAME = "config.yaml"
_CONFIG_DIR_NAME = "config"

# --- Singleton Pattern Implementation ---
_config_cache = None
_config_lock = threading.Lock()
# --- End Singleton Pattern ---

def load_config(config_path: str = f"{_CONFIG_DIR_NAME}/{_DEFAULT_CONFIG_FILENAME}"):
    global _config_cache
    # Check cache first (double-checked locking pattern)
    if _config_cache is not None:
 
        return _config_cache

    with _config_lock:
        # Check cache again inside lock in case another thread loaded it
        if _config_cache is not None:
            return _config_cache

        # --- Original loading logic starts here ---
        if yaml is None:
            print("PyYAML not installed, returning empty config.", file=os.sys.stderr)
            _config_cache = {} # Cache the empty dict
            return _config_cache

        potential_paths = [
            # Environment variable override - HIGHEST PRIORITY
            Path(os.environ.get("ALGOTUNE_CONFIG_PATH")) if os.environ.get("ALGOTUNE_CONFIG_PATH") else None,
            
            # Path relative to this loader.py file's parent (AlgoTuner/config/)
            Path(__file__).resolve().parent / _DEFAULT_CONFIG_FILENAME,

            # Path relative to this loader.py file's parent's parent (project root)
            Path(__file__).resolve().parent.parent / _CONFIG_DIR_NAME / _DEFAULT_CONFIG_FILENAME,

            Path(config_path),  # Path as provided (e.g., "config/config.yaml" relative to CWD)

            # Path relative to CWD if config_path was just the default "config/config.yaml"
            # This is somewhat redundant with Path(config_path) if config_path is default and CWD is stable
            Path.cwd() / _CONFIG_DIR_NAME / _DEFAULT_CONFIG_FILENAME,

            # Path if config_path was just "config.yaml" and it's in CWD
            Path.cwd() / _DEFAULT_CONFIG_FILENAME if config_path == _DEFAULT_CONFIG_FILENAME else None,
        ]

        loaded_config = None # Variable to store the successfully loaded config

        for p_path in potential_paths:
            if p_path is None:
                continue
            try:
                # Resolve to make sure it's absolute for consistent logging
                abs_path = p_path.resolve()
                if abs_path.exists() and abs_path.is_file():
                    print(f"Attempting to load config from: {abs_path}", file=os.sys.stderr)
                    with open(abs_path, "r") as file:
                        loaded_yaml = yaml.safe_load(file)
                        if loaded_yaml is not None:
                            print(f"Successfully loaded config from: {abs_path}", file=os.sys.stderr)
                            loaded_config = loaded_yaml # Store the loaded config
                            break # Stop searching once loaded
                        else:
                            print(f"Config file loaded but was empty: {abs_path}", file=os.sys.stderr)
                            # Treat empty file as empty config, but keep searching for non-empty
                            if loaded_config is None: # Only set if we haven't found a non-empty one yet
                                loaded_config = {} 
                else:
 
                    pass # Silently try next path
            except Exception as e:
                print(f"Error loading or parsing config file {abs_path}: {e}", file=os.sys.stderr)
                # Continue to try other paths

        if loaded_config is None: # If loop finished without loading anything valid
            print(f"Failed to load config from any potential paths. Defaulting to empty config. Searched: {[str(pp.resolve() if pp else 'None') for pp in potential_paths]}", file=os.sys.stderr)
            loaded_config = {}

        # Cache the final result (either loaded config or empty dict)
        _config_cache = loaded_config
        return _config_cache
