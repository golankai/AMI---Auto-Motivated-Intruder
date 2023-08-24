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
DATA_USED = "famous"

# Set seeds
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)

RESULTS_DIR = "./anon_grader/results/"

PRED_PATH = os.path.join(RESULTS_DIR, f"predictions_{SUDY_NUMBER}_{DATA_USED}_test.csv")
PRED_PATH2SAVE = os.path.join(RESULTS_DIR, f"predictions_{SUDY_NUMBER}_{DATA_USED}_test_PE.csv")
RESULTS_PATH = os.path.join(RESULTS_DIR, f"results_{SUDY_NUMBER}_{DATA_USED}_test.csv")
RESULTS_PATH2SAVE = os.path.join(RESULTS_DIR, f"results_{SUDY_NUMBER}_{DATA_USED}_test_PE.csv")
ERROR_FILES_DIR = f"./anon_grader/results/error_files_{SUDY_NUMBER}_{DATA_USED}"
if not os.path.exists(ERROR_FILES_DIR):
    os.makedirs(ERROR_FILES_DIR)

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

def _read_predictions_results(nm_samples, file_id="") -> pd.DataFrame:
    # Read the predictions from the PE phase
    if file_id == "" and os.path.exists(PRED_PATH2SAVE):
        predictions = pd.read_csv(PRED_PATH2SAVE, index_col=0)
        results = pd.read_csv(RESULTS_PATH2SAVE, index_col=0).to_dict(orient="index")
        nm_samples = len(predictions) if nm_samples == 0 else nm_samples
        if nm_samples != len(predictions):
            raise ValueError(  
                f"Number of samples in the predictions file is {len(predictions)}, but asked to run with {nm_samples}"
            )
    else:
        predictions = pd.read_csv(PRED_PATH, index_col=0)
        results = pd.read_csv(RESULTS_PATH, index_col=0).to_dict(orient="index")

        # Decrease the predictions to NUM_SAMPLES or to the one file
        if file_id != "":  # run on one file
            predictions = predictions[predictions["file_id"] == file_id]
        elif nm_samples != 0:  # run on NUM_SAMPLES
            predictions = predictions.sample(n=NUM_SAMPLES, random_state=SEED)
        else:  # run on all
            pass
    return predictions, results

# Get the score for each text
def _get_score_for_row(anon_text, de_anonymiser):
    for _ in range(5):
        response = de_anonymiser.re_identify(anon_text=anon_text)
        if response.get("status") == ResponseStatus.SUCCESS:
            return response.get("data").dict()["score"]
    return np.nan

def _get_self_const_score(anon_text, base_process_id):
    # Define the de-anonymizer
    de_anonymiser = DeAnonymizer(
        llm_name="chat-gpt",
        process_id=base_process_id,
        should_handle_data=True,
    )
    # Run 3 times to get the score
    responses = []
    for _ in range(3):
        responses.append(_get_score_for_row(anon_text, de_anonymiser))
    score = np.mean(responses)
    print("Self-Consistency score: ", score)
    return score

def _predict_pe(predictions: pd.DataFrame, process_ids: list[int]) -> pd.DataFrame:
    for process_id in process_ids:
        EXPERIMENT_NAME = get_exp_name(process_id)
        ERROR_FILE_PATH = f"{ERROR_FILES_DIR}/{EXPERIMENT_NAME}.csv"

        print(f"Running experiment: {EXPERIMENT_NAME}")

        if process_id == 1511:
            # Get the score for each text
            predictions[EXPERIMENT_NAME] = predictions["text"].apply(
                _get_self_const_score, args=(11,)
            )
        elif process_id == 1513:
            # Get the score for each text
            predictions[EXPERIMENT_NAME] = predictions["text"].apply(
                _get_self_const_score, args=(13,)
            )
        else:
            # Define the de-anonymizer
            de_anonymiser = DeAnonymizer(
                llm_name="chat-gpt",
                process_id=process_id,
                should_handle_data=True,
            )

            # Get the score for each text
            predictions[EXPERIMENT_NAME] = predictions["text"].apply(
                _get_score_for_row, args=(de_anonymiser,)
            )

        if file_id != "":  # got the printed results, no need to continue
            print("Predicted the given file, exiting...")
            sys.exit()

        error_files = de_anonymiser.get_error_files()
        if error_files is not None:
            error_files.to_csv(ERROR_FILE_PATH, index=False)
            print(
                "Save error files to csv successfully! file-name: ", ERROR_FILE_PATH
            )

        # Count the fails
        fails = len(predictions[predictions[EXPERIMENT_NAME].isna()])
        print(f"Failed {fails} times! Experiment {EXPERIMENT_NAME} Done!\n")
    
    # If ran processes 16x, calculate the normalized results for all Roles
    calc_roles_mean = any([process_id in [161, 162, 163, 164] for process_id in process_ids])
    if calc_roles_mean:
        roles_columns = [col for col in predictions.columns if col.startswith("Role")]
        if "Roles" in roles_columns:
            # Delete the Roles column from the predictions
            roles_columns.remove("Roles")
            predictions.drop(columns=["Roles"], inplace=True)
        predictions["Roles"] = np.nan
        predictions["Roles"] = predictions.apply(lambda row: np.mean([row[col] for col in roles_columns]), axis=1)
    
    # Save the predictions
    predictions.to_csv(PRED_PATH2SAVE)

    return predictions

    
def _calculate_results(predictions):
    # Prepare the predictions for the metrics
    preds_wo_none = predictions.dropna(subset=predictions.columns[5:])  

    # Calculate the overall results for each experiment
    results = {
        experiment: compute_metrics(
            (list(preds_wo_none[experiment]), list(preds_wo_none["human_rate"])),
            only_mse=False,
        )
        for experiment in preds_wo_none.columns[5:]
    }
    return results



if __name__ == "__main__":
    # Processes to run
    process_ids = [164] # [11, 111,  120, 121, 13, 14, 1511, 1513, 161, 162, 163, 164]
    NUM_SAMPLES = 0 # if 0, run on all
    file_id = ""  # if not empty, run on one file
    
    # Read the predictions and results
    predictions, results = _read_predictions_results(NUM_SAMPLES, file_id)
    predictions = _predict_pe(predictions, process_ids) # Comment out to run only the metrics

    # Calculate the overall results for each experiment
    results.update(_calculate_results(predictions))

    # Save the results
    results_df = pd.DataFrame.from_dict(results, orient="columns").T
    results_df.to_csv(RESULTS_PATH2SAVE)