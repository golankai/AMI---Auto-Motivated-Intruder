# import pandas as pd
# import os

# from de_anonymizer import DeAnonymizer



# def anonymise_text(dir: str, file_names: list[str]):
#     # Set up the de-anonimiser
#     de_anonymiser = DeAnonymizer(llm_name="chat-gpt", self_guide=True, verbose=True)

#     for file_name in file_names:
#         # Path to anonymized text
#         anon_text_path = f"{dir}/{file_name}"

#         # Read the data
#         with open(anon_text_path, "r") as f:
#             anon_text = f.read()    

#         answers = anon_process(anon_text, de_anonymiser)
#         # Run the de-anonimiser
#         answers = de_anonymiser.de_anonymise(anon_text)


# def anon_process(text: str, anonymizer: DeAnonymizer):





