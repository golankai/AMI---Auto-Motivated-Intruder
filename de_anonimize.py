from DeAnonimiser import DeAnonimiser

# Path to anonimised text
anon_text_path = "textwash_data/study2/person_descriptions/anon/adele_57.txt"

# Set up the de-anonimiser
de_anonimiser = DeAnonimiser()

# Define the text to de-anonimise
with open(anon_text_path, "r") as f:
    anon_text = f.read()

# Run the de-anonimiser
result = de_anonimiser.de_anonymise(anon_text)

# Print the result
print(result)