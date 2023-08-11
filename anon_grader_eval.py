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

models_names = [
    "study_1_famous_and_semi_class_epochs_1.pt",
    "study_1_famous_and_semi_class_and_11_epochs_5.pt"
]
# Set up environment
trained_models_path = f"./anon_grader/trained_models/"
data_dir = f"textwash_data/study{SUDY_NUMBER}/intruder_test/full_data_study.csv"
results_dir = "./anon_grader"

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

models_names = os.listdir(trained_models_path)
# Predict with all models
for model_name in models_names:
    # Create a copy of the data
    test_dataloader_copy = test_dataloader.copy()
    # Load the model
    logging.info(f"Loading model from {model_name}")

    model_path = os.path.join(trained_models_path, model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Prediction
    predictions = []
    for batch in test_dataloader_copy:
        with th.no_grad():
            regression_values = model(**batch).logits.squeeze().cpu().tolist()
        predictions.extend(regression_values)

    # Add predictions to the data
    data[f"model_{model_name}"] = predictions

    # Calculate the overall mse for each model
results = {
    model_name: compute_metrics((data[model_name], data["human_rate"]))["mse"]
    for model_name in data.columns[7:]
}

# Save predictions
data.to_csv(os.path.join(results_dir, "predictions.csv"))

# Save the results
results_df = pd.DataFrame.from_dict(results, orient="index", columns=["mse"])
results_df.to_csv(os.path.join(results_dir, "results.csv"))