# %%
import os
import logging
import pandas as pd
import numpy as np

import torch as th


from de_anonymizer.de_anonymizer import DeAnonymizer
from utils import compute_metrics, get_exp_name

# Processes to run
process_ids = [121]
# Define constants
TEMPERATURE = 0.5
SUDY_NUMBER = 1
NUM_SAMPLES = 5
DATA_USED = "famous"
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
# Get the score for each text
def _get_score_for_row(anon_text, de_anonymiser):
    de_anonymiser.re_identify(anon_text=anon_text)

# Run all the processes
for process_id in process_ids:
    EXPERIMENT_NAME = get_exp_name(process_id)
    print(f"Running experiment: {EXPERIMENT_NAME}")
    
    # Define the de-anonymizer
    de_anonymiser = DeAnonymizer(
        llm_name="chat-gpt", process_id=process_id, should_handle_data=should_handle_data, temperature=TEMPERATURE
    )

    # Get the score for each text
    predictions["text"].apply(_get_score_for_row, args=(de_anonymiser,))
    process_results = de_anonymiser.get_results()
    if "score" in process_results.columns:
        predictions[EXPERIMENT_NAME] = list(process_results["score"])

# Save the predictions
predictions.to_csv(PRED_PATH2SAVE)

# Calculate the rmse for this experiment
preds_we_none = predictions.dropna(subset=predictions.columns[5:])

results.update({
    experiment: compute_metrics((list(preds_we_none[experiment]), list(preds_we_none["human_rate"])), only_mse=False)
    for experiment in preds_we_none.columns[5:]
})

# Save the results
results_df = pd.DataFrame.from_dict(results, orient="columns").T
results_df.to_csv(RESULTS_PATH2SAVE)


# %%
