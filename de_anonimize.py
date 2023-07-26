import pandas as pd

from utils import read_data
from DeAnonimiser import DeAnonimiser

# Path to anonimised text
anon_text_path = "textwash_data/study2/person_descriptions/anon/adele_57.txt"

# Read the data
with open(anon_text_path, "r") as f:
    anon_text = f.read()
# df = read_data("textwash_data/study2/person_descriptions/anon")

# Set up the de-anonimiser
de_anonimiser = DeAnonimiser(llm="llama2")

# Run the de-anonimiser
result = de_anonimiser.de_anonymise(anon_text)

# Print the result
print(result)