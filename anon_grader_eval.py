import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch as th
from torch.utils.data import DataLoader, Dataset

from transformers import TrainingArguments, RobertaForSequenceClassification


from utils import prepare_grader_data, compute_metrics

# Define constants
SUDY_NUMBER = 1

data_used = "famous_and_semi"

# Set up environment
trained_models_path = f"./anon_grader/trained_models"
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
data = data[data["type"].isin(["famous", "semifamous"])]

# Preprocess the data
test_dataset = prepare_grader_data(data, SEED, DEVICE)['test']

# Create a DataLoader for the test dataset
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
)

# Predict with all models
for file in os.listdir(trained_models_path):

    if file.endswith(".pt"):
        model_path = os.path.join(trained_models_path, file)
        model = RobertaForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        model.eval()
        logging.info(f"Loading model from {model_path}")

    # Prediction
    predictions = []
    for batch in test_dataloader:
        with th.no_grad():
            regression_values = model(**batch).logits.squeeze().cpu().tolist()
        predictions.extend(regression_values)

    # Add predictions to the data
    data[f"model_{file}"] = predictions

# Calculate the overall mse for each model
results = {
    model_name: compute_metrics((data[model_name], data["human_rate"]))["mse"]
    for model_name in data.columns[7:]
}