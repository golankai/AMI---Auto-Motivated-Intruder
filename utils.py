import json
import os
import pandas as pd

from langchain.agents import load_tools
from langchain.llms import OpenAI, HuggingFaceHub, Cohere


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
        with open(os.path.join(dir, file), "r", encoding="utf-8") as f:
            anon_texts.append(f.read())

    df["file_name"] = files
    df["anon_text"] = anon_texts
    return df


def load_model(llm_name: str):
    """
    Load the LLM model.
    """
    match llm_name:
        case "flan-t5":
            repo_id = "google/flan-t5-xxl"
        case "llama2":
            repo_id = "meta-llama/Llama-2-70b-hf"
        case _:
            # raise an exception
            raise ValueError("llm name is not valid")
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 1024}
    )
    return llm


def load_google_search_tool():
    """
    Load the google search tool.
    """
    keys = get_local_keys()
    search = load_tools(["google-search"])[0]
    search.description = "A wrapper around Google Search. Useful for when you need to answer questions about current events or look for people who answer a specific charachteristic. Input should be a search query."
    return search
