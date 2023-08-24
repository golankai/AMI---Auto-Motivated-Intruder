import os
import logging
import pandas as pd
import numpy as np

import torch as th
from torch.utils.data import DataLoader

from transformers import RobertaForSequenceClassification

from clearml import Task


from utils import prepare_grader_data, compute_metrics, read_data_for_grader, predict

# Define constants
SUDY_NUMBER = 1
data_used = "famous"
EXPERIMENT_NAME = "eval_anon_grader_models"

task = Task.init(
    project_name="AMI_new",
    task_name=EXPERIMENT_NAME,
    task_type=Task.TaskTypes.testing,
)

# Set up environment
trained_models_path = f"./anon_grader/trained_models/"
RESULTS_DIR = "./anon_grader/results/"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
PRED_PATH = os.path.join(RESULTS_DIR, f"predictions_{SUDY_NUMBER}_{data_used}_val.csv")
PRED_PATH2SAVE = os.path.join(RESULTS_DIR, f"predictions_{SUDY_NUMBER}_{data_used}_test.csv")

RESULTS_PATH = os.path.join(RESULTS_DIR, f"results_{SUDY_NUMBER}_{data_used}_val.csv")
RESULTS_PATH2SAVE = os.path.join(RESULTS_DIR, f"results_{SUDY_NUMBER}_{data_used}_test.csv")

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

logging.info(f"Working on device: {DEVICE}")

# Cancel wandb logging
os.environ["WANDB_DISABLED"] = "true"


# Set seeds
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)


# Read the data
data = read_data_for_grader(SUDY_NUMBER, data_used, SEED)
val_data, test_data = data["val"], data["test"]
datasets = prepare_grader_data({"val": val_data, "test": test_data}, DEVICE)
val_dataset, test_dataset = datasets["val"], datasets["test"]

# Create a DataLoaders
val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
)

models_names = os.listdir(trained_models_path)

# Predict with all models
for model_name in models_names:
    predictions = predict(trained_models_path, val_dataloader, model_name, DEVICE)

    # Remove extension from model name
    val_data[f"model_{model_name[:-3]}"] = predictions

val_data.to_csv(PRED_PATH)
task.upload_artifact("Predictions df", artifact_object=val_data)

# Calculate the overall mse for each model
results = {
    model_name: compute_metrics((val_data[model_name], val_data["human_rate"]), only_mse=False)
    for model_name in val_data.columns[5:]
}

# Save the results
results_df = pd.DataFrame.from_dict(results, orient="columns").T
results_df.to_csv(RESULTS_PATH)
task.upload_artifact("Results df", artifact_object=results_df)

# Choose the best model
best_model = min(results, key=lambda x: results[x]["rmse"])
logging.info(f"Best model is {best_model}")

# Predict on test data
predictions = predict(trained_models_path, test_dataloader, best_model, DEVICE)
test_data[f"RoBERTa"] = predictions

# Calculate the metrics
results = compute_metrics((predictions, test_data["human_rate"]), only_mse=False)

# Save predictions and results on test data
test_data.to_csv(PRED_PATH2SAVE)
task.upload_artifact("Predictions on test", artifact_object=test_data)

results_df = pd.DataFrame.from_dict(results, orient="index").T
results_df.to_csv(RESULTS_PATH2SAVE)
task.upload_artifact("Results on test", artifact_object=results_df)