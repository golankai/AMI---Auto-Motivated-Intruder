import json




def get_local_keys():
    """Get local keys from a local keys.json file."""
    with open("keys.json", "r") as f:
        keys = json.load(f)
    return keys

def get_prompts_templates():
    """Get prompts template from a local prompts.json file."""
    with open("prompts.json", "r") as f:
        prompts = json.load(f)
    return prompts