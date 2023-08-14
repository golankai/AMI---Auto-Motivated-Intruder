import logging
import pandas as pd
import numpy as np

import torch as th


from clearml import Task


from utils import read_data_for_grader



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


# Read the data
data = read_data_for_grader(SUDY_NUMBER, DATA_USED, SEED)

train_data = data["train"]

# Choose text to use for the few shot
score_1 = train_data[train_data["file_id"] == "famous_398_d_1_10.txt"] 
score_0 = train_data[train_data["file_id"] == "semifamous_146_d_3_1.txt"] 
score_05 = train_data[train_data["file_id"] == "famous_138_d_1_4.txt"] 




