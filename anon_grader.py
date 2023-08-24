import os
import logging
import numpy as np

import torch as th

from transformers import TrainingArguments

from clearml import Task

from utils import train_grader_model, prepare_grader_data, read_data_for_grader

DEVICE = "cuda" if th.cuda.is_available() else "cpu"
RESULT_DIR = "./anon_grader/logs"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# Set seeds
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)

logging.info(f"Working on device: {DEVICE}")
# Cancel wandb logging
os.environ["WANDB_DISABLED"] = "true"

def run_training(epochs, data_used, layers_trained, study_number):

    # Define constants
    hyperparams = {
        "epochs": epochs,
        "data_used": data_used,
        "layers_trained": layers_trained,
        "study_number": study_number,
    }

    EXPERIMENT_NAME = f'study_{hyperparams[study_number]}_{hyperparams["data_used"]}_{hyperparams["layers_trained"]}_epochs_{hyperparams["epochs"]}'

    # Set up environment
    task = Task.init(
        project_name="AMI_new", task_name=EXPERIMENT_NAME)
    task.connect(hyperparams)

    trained_model_path = f"./anon_grader/trained_models/{EXPERIMENT_NAME}.pt"
    
    # Read the data
    data = read_data_for_grader(hyperparams["study_number"], hyperparams["data_used"], SEED, keep_more_than=3)

    datasets = prepare_grader_data(
        {
            "train": data["train"],
            "val": data["val"],
        },
        DEVICE,
    )

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=RESULT_DIR,
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

    # Train the model
    model = train_grader_model(
        datasets, training_args, hyperparams["layers_trained"], DEVICE
    )

    # save model
    th.save(model.state_dict(), trained_model_path)

    # Close the task
    task.close()


# Run training with all possible combinations of hyperparameters
epochs_lst = [5, 20]
data_used_lst = ["famous", "famous_and_semi", "all"]
layers_trained_lst = ["class", "class_and_11"]
study_number_lst = (1, 12)

for epochs in epochs_lst:
    for data_used in data_used_lst:
        for layers_trained in layers_trained_lst:
            for study_number in study_number_lst:
                run_training(epochs, data_used, layers_trained, study_number)

