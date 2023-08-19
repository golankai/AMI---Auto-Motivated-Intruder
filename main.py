import os
import random
from de_anonymizer.data_handler.data_handler import DataHandler
from de_anonymizer.de_anonymizer import DeAnonymizer

# Env parameters
process_id = 5

should_handle_data = True # handle dataFrame if True. Otherwise, just print the conversation.
single_text = False
run_all = False

study_number = 2

persona_name = "adele"
single_text_number = 576
text_lists = [43, 47, 57, 61, 97, 112, 147]#, 157, 178, 197, 201, 209, 216, 242, 271, 287, 297, 302, 323, 357, 366, 377, 397, 423, 442, 468, 491, 497, 503, 547, 558, 576]

result_csv_path = "results/study2/process5/t1"
texts_dir = f"textwash_data/study{study_number}/person_descriptions/anon"

############## ------------------------------- ##############


# Preserve the same order of texts between experiments
random.seed(42)

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
    de_anonymiser.re_identify_list(study_dir_path=texts_dir, file_names=texts_file_names, result_csv_path=result_csv_path)

