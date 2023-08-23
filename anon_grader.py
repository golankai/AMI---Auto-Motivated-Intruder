import os
import logging
import numpy as np

import torch as th

from transformers import TrainingArguments

from clearml import Task

from utils import train_grader_model, prepare_grader_data, read_data_for_grader



# Define constants#
hyperparams = {
    "epochs": 5,
    "data_used": "famous",
    "layers_trained": "class",
}

SUDY_NUMBER = 1

EXPERIMENT_NAME = f'study_{SUDY_NUMBER}_{hyperparams["data_used"]}_{hyperparams["layers_trained"]}_epochs_{hyperparams["epochs"]}'

# Set up environment
task = Task.init(project_name="AMI", task_name=EXPERIMENT_NAME, reuse_last_task_id=False)
task.connect(hyperparams)

trained_model_path = f"./anon_grader/trained_models/{EXPERIMENT_NAME}.pt"
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
data = read_data_for_grader(SUDY_NUMBER, hyperparams["data_used"], SEED)

datasets = prepare_grader_data({
        "train": data["train"],
        "val": data["val"],
    },
    DEVICE
)

# Train the model
model = train_grader_model(datasets, training_args, hyperparams["layers_trained"], DEVICE)

# save model
th.save(model.state_dict(), trained_model_path)
