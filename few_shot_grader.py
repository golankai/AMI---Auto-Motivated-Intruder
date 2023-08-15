# In[]
import logging
import pandas as pd
import numpy as np

import torch as th

from clearml import Task

from de_anonymizer.de_anonymizer import DeAnonymizer
from utils import read_data_for_grader, compute_metrics

# Define constants
DEBUG = True
SUDY_NUMBER = 1
DATA_USED = "famous_and_semi"
EXPERIMENT_NAME = f'few_shot_study_{SUDY_NUMBER}_{DATA_USED}'

# Set up environment
# task = Task.init(project_name="AMI", task_name=EXPERIMENT_NAME, reuse_last_task_id=False, task_typeTask.TaskTypes.inference)

data_dir = f"textwash_data/study{SUDY_NUMBER}/intruder_test/full_data_study.csv"
PRED_PATH = "./anon_grader/results/predictions_" + DATA_USED + ".csv"
RESULTS_PATH = "./anon_grader/results/results_" + DATA_USED + ".csv"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

logging.info(f'Working on device: {DEVICE}')

# Set seeds
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)


# Read the results of the models
predictions = pd.read_csv(PRED_PATH)
results = pd.read_csv(RESULTS_PATH)

# Read the data
data = read_data_for_grader(SUDY_NUMBER, DATA_USED, SEED)

train_data = data["train"]

# Choose text to use for the few shot
example_score_0 = train_data[train_data["file_id"] == "famous_398_d_1_10.txt"].text.values[0]
example_score_1 = train_data[train_data["file_id"] == "semifamous_146_d_3_1.txt"].text.values[0]
example_score_05 = train_data[train_data["file_id"] == "famous_138_d_1_4.txt"].text.values[0]

# decrease the prediction table size to 3, at random
predictions = predictions.sample(n=3, random_state=SEED)

# ChatGPT interaction

process_id = 3
should_handle_data = True 
study_number = 1

de_anonymiser = DeAnonymizer(
    llm_name="chat-gpt", process_id=process_id, should_handle_data=should_handle_data
)

def get_score_for_row(anon_text):
    de_anonymiser.re_identify(anon_text=anon_text, example_score_0=example_score_0, example_score_1=example_score_1, example_score_05=example_score_05)

predictions["text"].apply(get_score_for_row)

if should_handle_data:
    few_shot_preds = list(de_anonymiser.get_results()["score"])

# combine the results and the predictions
predictions["few_shot"] = few_shot_preds

few_shot_mse = compute_metrics((predictions["few_shot"], predictions["human_rate"]))["mse"]
results.loc[len(results)] = ["few_shot", few_shot_mse]

