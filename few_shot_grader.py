# %%
import os
import sys
import pandas as pd
import numpy as np

import torch as th


from de_anonymizer.de_anonymizer import DeAnonymizer
from utils import compute_metrics, get_exp_name
from conversations.conversation_handler import ResponseStatus

# Processes to run
process_ids = [13] # [11, 120, 121, 13, 14]
# Run on one file or all, if file_id is empty, run on all
# use with should_predict = False to run on one file, printing the results
# then write manually in the predictions csv and run again on all with should_predict = False
file_id = ""

# Predict or not
should_predict = True

# Define constants
TEMPERATURE = 0.5
SUDY_NUMBER = 1
NUM_SAMPLES = 5 # if 0, run on all
DATA_USED = "famous"
should_handle_data = True 

# Set up environment

PRED_PATH = f"./anon_grader/results/predictions_{SUDY_NUMBER}_{DATA_USED}.csv"
PRED_PATH2SAVE = f"./anon_grader/results/predictions_{SUDY_NUMBER}_{DATA_USED}_w_few_shot.csv"
RESULTS_PATH2SAVE = f"./anon_grader/results/results_{SUDY_NUMBER}_{DATA_USED}_w_few_shot.csv"
ERROR_FILES_DIR = f"./anon_grader/results/error_files_{SUDY_NUMBER}_{DATA_USED}"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

if not os.path.exists(ERROR_FILES_DIR):
    os.makedirs(ERROR_FILES_DIR)

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




# ChatGPT interaction
# Get the score for each text
def _get_score_for_row(anon_text, de_anonymiser):
    # Try 3 times to get the score
    for _ in range(3):
        response = de_anonymiser.re_identify(anon_text=anon_text)
        if response.get("status") == ResponseStatus.SUCCESS:
            return response.get("data").dict()["score"]
    return np.nan

# Run all the processes
if should_predict:
    # decrease the predictions to NUM_SAMPLES or to the one file
    if file_id != "": # run on one file
        predictions = predictions[predictions["file_id"] == file_id]
    elif NUM_SAMPLES != 0: # run on NUM_SAMPLES
        predictions = predictions.sample(n=NUM_SAMPLES, random_state=SEED)
    else: # run on all
        pass
    
    for process_id in process_ids:
        EXPERIMENT_NAME = get_exp_name(process_id)
        ERROR_FILE_PATH = f"{ERROR_FILES_DIR}/{EXPERIMENT_NAME}.csv"

        print(f"Running experiment: {EXPERIMENT_NAME}")
        
        # Define the de-anonymizer
        de_anonymiser = DeAnonymizer(
            llm_name="chat-gpt", process_id=process_id, should_handle_data=should_handle_data, temperature=TEMPERATURE
        )

        # Get the score for each text
        predictions[EXPERIMENT_NAME] = predictions["text"].apply(_get_score_for_row, args=(de_anonymiser,))

        if file_id != "": # got the printed results, no need to continue
            print("Predicted the given file, exiting...")
            sys.exit()

        if should_handle_data:
            error_files = de_anonymiser.get_error_files()
            if error_files is not None:
                error_files.to_csv(ERROR_FILE_PATH, index=False)
                print("Save error files to csv successfully! file-name: ", ERROR_FILE_PATH)

        # Count the fails
        fails =  len(predictions[predictions[EXPERIMENT_NAME].isna()])
        print(f"Failed {fails} times! Experiment {EXPERIMENT_NAME} Done!\n")

    # Save the predictions
    predictions.to_csv(PRED_PATH2SAVE)

# Calculate the rmse for this experiment
preds_wo_none = predictions.dropna(subset=predictions.columns[5:])

results.update({
    experiment: compute_metrics((list(preds_wo_none[experiment]), list(preds_wo_none["human_rate"])), only_mse=False)
    for experiment in preds_wo_none.columns[5:]
})

# Save the results
results_df = pd.DataFrame.from_dict(results, orient="columns").T
results_df.to_csv(RESULTS_PATH2SAVE)


# %%
