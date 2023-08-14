import json
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tqdm
from typing import List, Dict, Any, Optional, Union, Tuple
from re import sub, match


#from langchain.agents import load_tools
#from langchain.llms import HuggingFaceHub, Cohere, OpenAI
#from langchain.chat_models import ChatOpenAI


import torch as th
from torch.optim import AdamW
from torch.utils.data import Dataset
from datasets import DatasetDict

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup


def get_local_keys():
    """Get local keys from a local keys.json file."""
    with open("keys.json", "r") as f:
        keys = json.load(f)
    return keys


def read_data(dir: str):
    """
    Read data from a directory to a panda DataFrame.
    """

    files = os.listdir(dir)
    files.remove(".DS_Store")
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
    # match llm_name:
    #     case "chat-gpt":
    #         return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=512)
    #     case "flan-t5":
    #         repo_id = "declare-lab/flan-alpaca-large"
    #     case "llama2":
    #         repo_id = "meta-llama/Llama-2-70b-chat-hf"
    #     case _:
    #         # raise an exception
    #         raise ValueError("llm name is not valid")

    # llm = HuggingFaceHub(
    #     repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 512}
    # )
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=512)


def load_google_search_tool():
    """
    Load the google search tool.
    """
    keys = get_local_keys()
    os.environ["GOOGLE_API_KEY"] = keys["google_api_key"]
    os.environ["GOOGLE_CSE_ID"] = keys["google_cse_id"]
    search = load_tools(["google-search"])[0]
    search.description = "A wrapper around Google Search. Useful for when you need to answer questions about current events or look for people who answer a specific charachteristic. Input should be a search query."
    return search


######################################
###   Grader Functions  
######################################

from torch.utils.data import Dataset


class GraderDataset(Dataset):
    def __init__(self, inputs, labels, device):
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input_ids = self.input_ids[index].squeeze().to(self.device)
        attention_mask = self.attention_mask[index].squeeze().to(self.device)
        labels = th.tensor(self.labels[index]).squeeze().to(self.device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_grader_model(
    datasets: Dict[str, GraderDataset],
    training_args: TrainingArguments,
    layers_trained: str,
    device,
):
    """
    Train the grader model.
    :param datasets: the data to train and validate on
    :param seed: the seed for the random state
    :param training_args: the training arguments
    :param trained_model_path: the path to save the trained model
    :param layers_trained: the layers to train
    :param device: the device to train on
    :return: the trained model and tokenizer
    """
    # Extract the train and validation datasets
    train_dataset, val_dataset = datasets["train"], datasets["val"]

    # Load the model
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=1
    ).to(device)

    # Set the training optimizer
    params = model.named_parameters()
    top_layer_params = []
    for name, para in params:
        # require grad only for needed layers
        pattern = get_layer_pattern(layers_trained)
        if match(pattern, name):
            para.requires_grad = True
            top_layer_params.append(para)
        else:
            para.requires_grad = False

    optimizer = AdamW(top_layer_params, lr=1e-4)

    # Set the scheduler
    total_steps = len(train_dataset) * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Instantiate the Trainer class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
    )

    # Train the model with tqdm progress bar
    with tqdm.trange(training_args.num_train_epochs, desc="Epoch") as t:
        for epoch in t:
            trainer.train()
            t.set_description(f"Epoch {epoch}")

    return model


def prepare_grader_data(data_splits: Dict[str, pd.DataFrame], device) -> DatasetDict:
    """
    Create train, validation, test datasets and tokenizer for the grader model.
    :param data: the data splits to process
    :param device: the device to train on
    :return: the trained model and tokenizer
    """
    # Load pre-trained RoBERTa tokenizer and model
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    datasets = {}
    for split, data in data_splits.items():
        # Preprocessing
        texts = data["text"].tolist()
        labels = data["human_rate"].tolist()

        
        # Tokenize input texts
        encodings = tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        )

        # Create dataset objects
        datasets[split] = GraderDataset(encodings, labels, device)

    return DatasetDict(datasets)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions, squared=False)
    return {"mse": mse}

def choose_data(data: pd.DataFrame, data_used : str) -> pd.DataFrame:
    '''
    Choose the data to use based on the types.
    :param data: the data to choose from.
    :param data_used: the type of data to use.
    :return: the data to use.
    '''
    if data_used == "all":
        return data
    elif data_used == "famous":
        return data[data["type"].isin(["famous"])]
    elif data_used == "famous_and_semi":
        return data[data["type"].isin(["famous", "semifamous"])]
    else:
        raise Exception("Invalid data type.")
    
def get_layer_pattern(layers_trained: str) -> str:
    '''
    Get the pattern for the layers to train.
    :param layers_trained: the layers to train.
    :return: the pattern for the layers to train.
    '''
    if layers_trained == "class":
        return r"classifier.*"
    elif layers_trained == "class_and_11":
        return r'classifier.*|roberta.encoder.layer.11.*'
    else:
        raise Exception("Invalid layers trained.")
