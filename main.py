import pandas as pd
import os

from de_anonymizer.de_anonymizer import DeAnonymizer

# Env parameters
process_id = 1
save_to_csv = False
single_text = False
run_all = False
study_number = 2
persona_name = "adele"
single_text_number = 576
text_lists = [43, 47] #[43, 47, 57, 61, 97, 112, 147, 157, 178, 197, 201, 209, 216, 242, 271, 287, 297, 302, 323, 357, 366, 377, 397, 423, 442, 468, 491, 497, 503, 547, 558, 576]
result_csv_path = "pre-study/adele/forth"

texts_dir = f"textwash_data/study{study_number}/person_descriptions/anon"

texts_file_names = []
if run_all:
    all_files_in_dir = os.listdir(texts_dir)
    texts_file_names = [file_name for file_name in all_files_in_dir]
else:
    texts_file_names = [
        f"{persona_name}_{text_number}.txt" for text_number in text_lists
    ]

de_anonymiser = DeAnonymizer(
    llm_name="chat-gpt", process_id=process_id, self_guide=True, verbose=True
)

if single_text:
    with open(f"{texts_dir}/{persona_name}_{single_text_number}.txt", "r") as f:
        anon_text = f.read()
    response = de_anonymiser.re_identify(anon_text)
    print(response)
else:
    df = de_anonymiser.re_identify_list(study_dir_path=texts_dir, file_names=texts_file_names, save_to_csv=save_to_csv)
    if df is not None:
        df.to_csv(f"results/{result_csv_path}.csv", index=False)

