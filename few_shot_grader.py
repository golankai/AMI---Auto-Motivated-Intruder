import os
import logging
import pandas as pd
import numpy as np

import torch as th

from clearml import Task

from de_anonymizer.de_anonymizer import DeAnonymizer
from utils import read_data_for_grader, compute_metrics, get_process_id

# Define constants
SUDY_NUMBER = 1
NUM_SAMPLES = 7
DATA_USED = "famous"
EXPERIMENT_NAME = "zero_shot"
process_id = get_process_id(EXPERIMENT_NAME)
should_handle_data = True 

# Set up environment

PRED_PATH = f"./anon_grader/results/predictions_{SUDY_NUMBER}_{DATA_USED}.csv"
PRED_PATH2SAVE = f"./anon_grader/results/predictions_{SUDY_NUMBER}_{DATA_USED}_w_few_shot.csv"
RESULTS_PATH2SAVE = f"./anon_grader/results/results_{SUDY_NUMBER}_{DATA_USED}_w_few_shot.csv"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

logging.info(f'Working on device: {DEVICE}')

# Set seeds
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)

# If alreday have predictions with few-shot, read them
if os.path.exists(PRED_PATH2SAVE):
    predictions = pd.read_csv(PRED_PATH2SAVE, index_col=0)
    assert len(predictions) == NUM_SAMPLES, "The number of samples is not as expected"
    results = pd.read_csv(RESULTS_PATH2SAVE, index_col=0).to_dict(orient="index")
else: 
    # Read the predictions from the models
    predictions = pd.read_csv(PRED_PATH, index_col=0)
    # Calculate the scores for each model
    results = {
        model_name: compute_metrics((list(predictions[model_name]), list(predictions["human_rate"])), only_mse=False)
        for model_name in predictions.columns[5:]
    }
    # Keep the best model, based on the mse
    best_model = min(results, key=lambda x: results[x]["rmse"])
    results = {
        "data": compute_metrics((list(predictions["human_rate"]), list(predictions["human_rate"])), only_mse=False),
        "RoBERTa": results[best_model]
        }
    # Keep the predictions of the best model only
    predictions = predictions[["type", "file_id" , "name",  "text" , "human_rate", best_model]].rename(columns={best_model: "RoBERTa"})

# decrease the predictions to NUM_SAMPLES
predictions = predictions.sample(n=NUM_SAMPLES, random_state=SEED)


# ChatGPT interaction

# Define the de-anonymizer
de_anonymiser = DeAnonymizer(
    llm_name="chat-gpt", process_id=process_id, should_handle_data=should_handle_data
)

# Get the score for each text
def _get_score_for_row(anon_text):
    de_anonymiser.re_identify(anon_text=anon_text)

predictions["text"].apply(_get_score_for_row)
predictions[EXPERIMENT_NAME] = list(de_anonymiser.get_results()["score"])

# Save the predictions
predictions.to_csv(PRED_PATH2SAVE)

# Calculate the rmse for this experiment
results[EXPERIMENT_NAME] = compute_metrics((list(predictions[EXPERIMENT_NAME]), list(predictions["human_rate"])), only_mse=False)

# Save the results
results_df = pd.DataFrame.from_dict(results, orient="columns").T
results_df.to_csv(RESULTS_PATH2SAVE)

