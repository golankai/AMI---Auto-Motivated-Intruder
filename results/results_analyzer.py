from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd

from personas.personas import Persona

PERSONA_COL_NAME = "Persona"

class ResultsAnalyzer:
    def __init__(self, csv_results_path: str, predicted_column_name="name"):
        self.df_results = pd.read_csv(csv_results_path)
        self.predicted_column_name = predicted_column_name

        # Add persona column
        self.add_persona_column()
        # Add is_right column
        self.add_is_right_column()
        
        self.df_results.to_csv
    

    def add_persona_column(self):
        def get_persona_value_by_file(file_name: str):
            # Inputs:
            # :file_name: str - in the format of: <persona_family_name>_<text_number>.txt
            # We still using the enum to get error if something unexpected happens

            family_name = file_name.split("_")[0].capitalize()
            return Persona[family_name].value
        
        self.df_results[PERSONA_COL_NAME] = self.df_results.apply(lambda row: get_persona_value_by_file(row["File"]), axis=1)


    def add_is_right_column(self):
        def is_right(predicted_name: str, persona_value: str):
            persona = Persona[persona_value.capitalize()]
            return predicted_name in persona.get_optional_names()

        self.df_results["is_right"] = self.df_results.apply(lambda row: is_right(row[self.predicted_column_name], row[PERSONA_COL_NAME]), axis=1)
    

    def get_results_data_frame(self):
        return self.df_results

    def persona_accuracy(self, persona: Persona):
        df_persona = self.df_results[persona.value == self.df_results[PERSONA_COL_NAME]]
        if df_persona.shape[0] == 0:
            raise Exception(f"No results for persona {persona}")
        
        return df_persona["is_right"].mean()

    
