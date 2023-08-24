import os
import sys
import pandas as pd
import numpy as np

import torch as th


from de_anonymizer.de_anonymizer import DeAnonymizer
from utils import compute_metrics, get_exp_name
from conversations.conversation_handler import ResponseStatus

# Define constants
SUDY_NUMBER = 1
NUM_SAMPLES = 0  # if 0, run on all
DATA_USED = "famous"

RESULTS_DIR = "./anon_grader/results/"
PRED_PATH = os.path.join(RESULTS_DIR, f"predictions_{SUDY_NUMBER}_{DATA_USED}_val.csv")
RESULTS_PATH = os.path.join(RESULTS_DIR, f"results_{SUDY_NUMBER}_{DATA_USED}_val.csv")

PRED_PATH2SAVE = os.path.join(RESULTS_DIR, f"predictions_{SUDY_NUMBER}_{DATA_USED}_test.csv")
RESULTS_PATH2SAVE = os.path.join(RESULTS_DIR, f"results_{SUDY_NUMBER}_{DATA_USED}_test.csv")
ERROR_FILES_DIR = f"./anon_grader/results/error_files_{SUDY_NUMBER}_{DATA_USED}"
if not os.path.exists(ERROR_FILES_DIR):
    os.makedirs(ERROR_FILES_DIR)

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

def _read_predictions() -> pd.DataFrame:
    # Read the predictions from the PE phase
    if os.path.exists(PRED_PATH2SAVE):
        predictions = pd.read_csv(PRED_PATH2SAVE, index_col=0)
        results = pd.read_csv(RESULTS_PATH2SAVE, index_col=0).to_dict(orient="index")
    else: # read the predictions of the models
        # Read the results of the models on val data
        results = pd.read_csv(RESULTS_PATH, index_col=0).to_dict(orient="index")
        # Choose the best model
        best_model = min(results, key=lambda x: results[x]["rmse"])
        # Read the predictions of the best model
        predictions = pd.read_csv(PRED_PATH, index_col=0, usecols=["type", "file_id", "name", "text", "human_rate", best_model])
        # Keep the predictions of the best model only
        predictions = predictions.rename(columns={best_model: "RoBERTa"})

ROLE_NR = 1 # if working with process 16
calc_roles_mean = False # if working with process 16 and want to calculate the mean of the roles

# Run on one file or all, if file_id is empty, run on all
# use with should_predict = True to run on one file, printing the results
# then write manually in the predictions csv and run again on all with should_predict = False
file_id = ""

# Predict or not
should_predict = True




# Set seeds
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)




# ChatGPT interaction
# Get the score for each text
def _get_score_for_row(anon_text, de_anonymiser):
    for _ in range(10):
        response = de_anonymiser.re_identify(anon_text=anon_text)
        if response.get("status") == ResponseStatus.SUCCESS:
            return response.get("data").dict()["score"]
    return np.nan


def _get_self_const_score(anon_text, base_process_id):
    # Define the de-anonymizer
    de_anonymiser = DeAnonymizer(
        llm_name="chat-gpt",
        process_id=base_process_id,
        should_handle_data=should_handle_data,
    )
    # Run 3 times to get the score
    responses = []
    for _ in range(3):
        responses.append(_get_score_for_row(anon_text, de_anonymiser))
    score = np.mean(responses)
    print("Self-Consistency score: ", score)
    return score


# Run all the processes
if should_predict:
    # decrease the predictions to NUM_SAMPLES or to the one file
    if file_id != "":  # run on one file
        predictions = predictions[predictions["file_id"] == file_id]
    elif NUM_SAMPLES != 0:  # run on NUM_SAMPLES
        predictions = predictions.sample(n=NUM_SAMPLES, random_state=SEED)
    else:  # run on all
        pass

    for process_id in process_ids:
        EXPERIMENT_NAME = get_exp_name(process_id)
        if process_id == 16:
            EXPERIMENT_NAME += str(ROLE_NR)
        ERROR_FILE_PATH = f"{ERROR_FILES_DIR}/{EXPERIMENT_NAME}.csv"

        print(f"Running experiment: {EXPERIMENT_NAME}")

        if process_id == 1511:
            # Get the score for each text
            predictions[EXPERIMENT_NAME] = predictions["text"].apply(
                _get_self_const_score, args=(11,)
            )
            continue
        elif process_id == 1513:
            # Get the score for each text
            predictions[EXPERIMENT_NAME] = predictions["text"].apply(
                _get_self_const_score, args=(13,)
            )
            continue
        else:
            pass

        # Define the de-anonymizer
        de_anonymiser = DeAnonymizer(
            llm_name="chat-gpt",
            process_id=process_id,
            should_handle_data=should_handle_data,
        )

        # Get the score for each text
        predictions[EXPERIMENT_NAME] = predictions["text"].apply(
            _get_score_for_row, args=(de_anonymiser,)
        )

        if file_id != "":  # got the printed results, no need to continue
            print("Predicted the given file, exiting...")
            sys.exit()

        if should_handle_data:
            error_files = de_anonymiser.get_error_files()
            if error_files is not None:
                error_files.to_csv(ERROR_FILE_PATH, index=False)
                print(
                    "Save error files to csv successfully! file-name: ", ERROR_FILE_PATH
                )

        # Count the fails
        fails = len(predictions[predictions[EXPERIMENT_NAME].isna()])
        print(f"Failed {fails} times! Experiment {EXPERIMENT_NAME} Done!\n")

    # Save the predictions
    predictions.to_csv(PRED_PATH2SAVE)

# Calculate the rmse for this experiment
preds_wo_none = predictions.dropna(subset=predictions.columns[5:])

results.update(
    {
        experiment: compute_metrics(
            (list(preds_wo_none[experiment]), list(preds_wo_none["human_rate"])),
            only_mse=False,
        )
        for experiment in preds_wo_none.columns[5:]
    }
)

# If predictiones have Role_x experiments, calculate the normalized results for all Roles
if calc_roles_mean:
    roles_columns = [col for col in results.keys() if col.startswith("Role")]

    predictions["Roles"] = predictions.apply(lambda row: np.mean([row[col] for col in roles_columns]), axis=1)
    # Save the predictions
    predictions.to_csv(PRED_PATH2SAVE)
    results.update(
        {
            "Roles": compute_metrics(
                (list(predictions["Roles"]), list(predictions["human_rate"])),
                only_mse=False,
            )
        }
    )
    
# Save the results
results_df = pd.DataFrame.from_dict(results, orient="columns").T
results_df.to_csv(RESULTS_PATH2SAVE)


if __name__ == "__main__":
    # Processes to run
    process_ids = [11, 111,  120, 121, 13, 14, 1511, 1513, 161, 162, 163, 163] # [11, 111,  120, 121, 13, 14, 1511, 1513, 161, 162, 163, 163]
    predictions = _read_predictions()