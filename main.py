import os
import random
from de_anonymizer.de_anonymizer import DeAnonymizer
from evaluation.experiment_evaluation import ExperimentEvaluation
from personas.personas import Persona

# Env parameters
study_number = 2
process_id = 5
experiment_number = 1

should_handle_data = True # handle dataFrame if True. Otherwise, just print the conversation.
single_text = False
run_all = False

persona_name = "adele"
single_text_number = 576
text_lists = [43, 47, 57, 61, 97, 112, 147]#, 157, 178, 197, 201, 209, 216, 242, 271, 287, 297, 302, 323, 357, 366, 377, 397, 423, 442, 468, 491, 497, 503, 547, 558, 576]

result_type = "ami" # score
############## ------------------------------- ##############

texts_dir = f"textwash_data/study{study_number}/person_descriptions/anon"

result_base_path = f"results/{result_type}/study{study_number}/process{process_id}/experiment{experiment_number}"
results_paths = {
    "raw": f"{result_base_path}/raw_results.csv",
    "error_files_raw": f"{result_base_path}/raw_error_files.csv",
    "process": f"{result_base_path}/processed.csv",
    "evaluation": f"{result_base_path}/evaluation.json",
}


# Preserve the same order of texts between experiments
random.seed(42)

def intruder():
    texts_file_names = []
    if run_all:
        all_files = os.listdir(texts_dir)
        random.shuffle(all_files)
        texts_file_names = all_files
        print("Run all files! ", len(texts_file_names))
    else:
        texts_file_names = [
            f"{persona_name}_{text_number}.txt" for text_number in text_lists
        ]

    de_anonymiser = DeAnonymizer(
        llm_name="chat-gpt", process_id=process_id, self_guide=True, verbose=True, should_handle_data=should_handle_data
    )

    if single_text:
        with open(f"{texts_dir}/{persona_name}_{single_text_number}.txt", "r") as f:
            anon_text = f.read()
        de_anonymiser.re_identify(anon_text)
    else:
        de_anonymiser.re_identify_list(
            study_dir_path=texts_dir, 
            file_names=texts_file_names, 
            result_path=results_paths["raw"],
            error_files_path=results_paths["error_files_raw"]
            )


def analyzer():
    result_analyzer = ExperimentEvaluation(raw_results_path=results_paths["raw"])
    result_analyzer.process_results(path=results_paths["process"])
    result_analyzer.evaluate(path=results_paths["evaluation"])


analyzer()
