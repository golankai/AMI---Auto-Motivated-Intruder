import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch as th
from torch.utils.data import DataLoader, Dataset

from transformers import TrainingArguments, RobertaForSequenceClassification

from clearml import Task


from utils import prepare_grader_data, compute_metrics, read_data_for_grader

# Define constants
SUDY_NUMBER = 12
data_used = "famous_and_semi"
EXPERIMENT_NAME = f'eval_study_{SUDY_NUMBER}_{data_used}'

task = Task.init(project_name="AMI", task_name=EXPERIMENT_NAME, reuse_last_task_id=False, task_type=Task.TaskTypes.testing)

# Set up environment
trained_models_path = f"./anon_grader/trained_models/"
PRED_PATH = f"./anon_grader/results/predictions_{SUDY_NUMBER}_{data_used}.csv"
RESULTS_PATH = f"./anon_grader/results/results_{SUDY_NUMBER}_{data_used}.csv"

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

logging.info(f'Working on device: {DEVICE}')

# Cancel wandb logging
os.environ["WANDB_DISABLED"] = "true"


# Set seeds
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)


# Read the data
test_data = read_data_for_grader(data_used, SEED)['test']

test_dataset = prepare_grader_data({"test": test_data}, DEVICE)['test']

# Create a DataLoader for the test dataset
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)

models_names = os.listdir(trained_models_path)
# take only models trained on the study and data used
models_names = [model_name for model_name in models_names if f"study_{SUDY_NUMBER}" in model_name and data_used in model_name]

# Predict with all models
for model_name in models_names:
    predictions = []

    # Load the model
    logging.info(f"Loading model from {model_name}")

    model_path = os.path.join(trained_models_path, model_name)
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1).to(DEVICE)
    model.load_state_dict(th.load(model_path))
    model.eval()

    # Prediction
    for batch in test_dataloader:
        with th.no_grad():
            
            outputs = model(**batch)          
            regression_values = outputs["logits"].squeeze().cpu().tolist()
            
        predictions.extend(regression_values)

    # Clip predictions to [0, 1]
    predictions = np.clip(predictions, 0, 1)

    # Add predictions to the data

    # Remove extension from model name
    test_data[f"model_{model_name[:-3]}"] = predictions

test_data.to_csv(PRED_PATH)
task.upload_artifact("Predictions df", artifact_object=test_data)

# Calculate the overall mse for each model
results = {
    model_name: compute_metrics((test_data[model_name], test_data["human_rate"]))["mse"]
    for model_name in test_data.columns[5:]
}

# Save the results
results_df = pd.DataFrame.from_dict(results, orient="index", columns=["mse"])
results_df.to_csv(RESULTS_PATH)
task.upload_artifact("Results df", artifact_object=results_df)
