# Imports
import json

# General imports

# LLM imports

# Local imports

def get_local_keys():
    """Get local keys from a local keys.json file."""
    with open("keys.json", "r") as f:
        keys = json.load(f)
    return keys