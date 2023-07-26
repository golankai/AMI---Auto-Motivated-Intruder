import json
import os
import pandas as pd

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

def read_data(dir: str) -> pd.DataFrame:
    """
    Read data from a directory to a panda DataFrame.
    """

    files = os.listdir(dir)
    df = pd.DataFrame(columns=["file_name", "anon_text"])
    anon_texts = []
    for file in files:
        with open(os.path.join(dir, file), "r", encoding='utf-8') as f:
            anon_texts.append(f.read())

    df["file_name"] = files
    df["anon_text"] = anon_texts
    return df
    