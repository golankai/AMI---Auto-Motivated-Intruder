import pandas as pd

from utils import read_data
from DeAnonimiser import DeAnonimiser

# Path to anonimised text
anon_text_path = "textwash_data/study2/person_descriptions/anon/adele_57.txt"
# anon_texts_path = "textwash_data/study2/person_descriptions/anon"

# Read the data
with open(anon_text_path, "r") as f:
    anon_text = f.read()
# df = read_data("textwash_data/study2/person_descriptions/anon")
# df = df.head(10)

# Set up the de-anonimiser
de_anonimiser = DeAnonimiser(llm="flan-t5", verbose=True)

# Run the de-anonimiser
result = de_anonimiser.de_anonymise(anon_text)
# df['de_anon_result'] = df.apply(lambda row: de_anonimiser.de_anonymise(row["anon_text"]), axis=1)

# Print the result
print(result)
