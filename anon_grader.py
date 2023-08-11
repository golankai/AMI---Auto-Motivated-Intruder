import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch as th

from transformers import TrainingArguments

from clearml import Task


from utils import train_grader_model, prepare_grader_data, choose_data



# Define constants#
hyperparams = {
    "epochs": 20,
    "data_used": "famous_and_semi",
    "layers_trained": "class",
}

DEBUG = True
SUDY_NUMBER = 1

EXPERIMENT_NAME = f'study_{SUDY_NUMBER}_{hyperparams["data_used"]}_{hyperparams["layers_trained"]}_epochs_{hyperparams["epochs"]}'

# Set up environment
task = Task.init(project_name="AMI", task_name=EXPERIMENT_NAME, reuse_last_task_id=False)
task.connect(hyperparams)

trained_model_path = f"./anon_grader/trained_models/{EXPERIMENT_NAME}.pt"
data_dir = f"textwash_data/study{SUDY_NUMBER}/intruder_test/full_data_study.csv"
results_dir = "./anon_grader/logs"

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

logging.info(f'Working on device: {DEVICE}')
# Cancel wandb logging
os.environ["WANDB_DISABLED"] = "true"


# Set seeds
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)


# Set up the training arguments
training_args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=hyperparams["epochs"],
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=1,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="tensorboard",
    dataloader_pin_memory=False,
)


# Read the data
columns_to_read = ["type", "text", "file_id", "name", "got_name_truth_q2"]
raw_data = pd.read_csv(data_dir, usecols=columns_to_read)


# Aggregate by file_id and calculate the rate of re-identification
data = (
    raw_data.groupby(["type", "file_id", "name", "text"])
    .agg({"got_name_truth_q2": "mean"})
    .reset_index()
)
data.rename(columns={"got_name_truth_q2": "human_rate"}, inplace=True)

# Define population to use
data = choose_data(data, hyperparams["data_used"])

# Preprocess the data
datasets = prepare_grader_data(data, SEED, DEVICE)

# Train the model
model = train_grader_model(datasets, training_args, hyperparams["layers_trained"], DEVICE)

# save model
th.save(model.state_dict(), trained_model_path)
