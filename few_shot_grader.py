# %%
import logging
import pandas as pd
import numpy as np

import torch as th

from clearml import Task

from de_anonymizer.de_anonymizer import DeAnonymizer
from utils import read_data_for_grader, compute_metrics
# %%
# Define constants
SUDY_NUMBER = 12
NUM_SAMPLES = 5
DATA_USED = "famous_and_semi"
EXPERIMENT_NAME = f'few_shot_study_{SUDY_NUMBER}_{DATA_USED}'

# task = Task.init(project_name="AMI", task_name=EXPERIMENT_NAME, reuse_last_task_id=False, task_type=Task.TaskTypes.testing)
# Set up environment

PRED_PATH = "./anon_grader/results/predictions_" + SUDY_NUMBER + "_" + DATA_USED + ".csv"
PRED_PATH2SAVE = "./anon_grader/results/predictions_" + SUDY_NUMBER + "_" + DATA_USED+ "_w_few_shot.csv"
RESULTS_PATH = "./anon_grader/results/results_" + SUDY_NUMBER + "_" + DATA_USED + "_w_few_shot.csv"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

logging.info(f'Working on device: {DEVICE}')

# Set seeds
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)

# %%
# Read the results of the models
predictions = pd.read_csv(PRED_PATH, index_col=0)


# Read the data
data = read_data_for_grader(DATA_USED, SEED)

train_data = data["train"]

# Choose text to use for the few shot
example_score_1 = train_data[train_data["file_id"] == "famous_398_d_1_10.txt"].text.values[0]
example_score_0 = train_data[train_data["file_id"] == "semifamous_146_d_3_1.txt"].text.values[0]
example_score_05 = train_data[train_data["file_id"] == "famous_138_d_1_4.txt"].text.values[0]
# %%

# decrease the prediction table size to 3, at random
predictions = predictions.sample(n=NUM_SAMPLES)

# Calculate the scores for each model
results = {
    model_name: compute_metrics((list(predictions[model_name]), list(predictions["human_rate"])), only_mse=False)
    for model_name in predictions.columns[5:]
}

# Keep the best model, based on the mse
best_model = min(results, key=lambda x: results[x]["mse"])
results = {"best_RoBERTa": results[best_model]}

# Keep the predictions of the best model only

predictions = predictions[["type", "file_id", "name", "text", "human_rate", best_model]].rename(columns={best_model: "best_RoBERTa"}, inplace=True)
# %%

# ChatGPT interaction
process_id = 3
should_handle_data = True 

# Define the de-anonymizer
de_anonymiser = DeAnonymizer(
    llm_name="chat-gpt", process_id=process_id, should_handle_data=should_handle_data
)

# Get the score for each text
def get_score_for_row(anon_text):
    de_anonymiser.re_identify(anon_text=anon_text, example_score_0=example_score_0, example_score_1=example_score_1, example_score_05=example_score_05)
# %%

predictions["text"].apply(get_score_for_row)
predictions["few_shot"] = list(de_anonymiser.get_results()["score"])
# %%

# task.upload_artifact("Predictions df with few-shot", artifact_object=predictions)
predictions.to_csv(PRED_PATH2SAVE)

# Calculate the mse for few-shot
results["few_shot"] = compute_metrics((list(predictions["few_shot"]), list(predictions["human_rate"])), only_mse=False)

# Save the results
results_df = pd.DataFrame.from_dict(results, orient="index", columns=["mse", "avd_pred"])
results_df.to_csv(RESULTS_PATH)
# task.upload_artifact("Results df with few-shot", artifact_object=results_df)

# %%
