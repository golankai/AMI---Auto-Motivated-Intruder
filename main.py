import pandas as pd
import os

from de_anonymizer import DeAnonymizer


run_all = False
study_number = 2
persona_name = "adele"
text_lists = [43, 47, 57, 61, 97, 112, 147, 157, 178, 197]

texts_dir = f"textwash_data/study{study_number}/person_descriptions/anon"
texts_file_names = []
if run_all:
    all_files_in_dir = os.listdir(texts_dir)
    texts_file_names = [file_name for file_name in all_files_in_dir]
else:
    texts_file_names = [f"{persona_name}_{text_number}.txt" for text_number in text_lists]


de_anonymiser = DeAnonymizer(llm_name="chat-gpt", self_guide=True, verbose=True)

with open(f"{texts_dir}/{texts_file_names[0]}", "r") as f:
    anon_text = f.read()

de_anonymiser.de_anonymise(anon_text)
