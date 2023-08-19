from dataclasses import dataclass
from enum import Enum
import pandas as pd

from results.personas import Persona

PERSONA_COL_NAME = "Persona"

class ResultsAnalyzer:
    def __init__(self, csv_results_path: str, name_of_predicted_column="name"):
        self.df_results = pd.read_csv(csv_results_path)
        
        # Add persona column
        self.df_results[PERSONA_COL_NAME] = self.df_results.apply(lambda row: row["File"].split("_")[0], axis=1)
        # Fix Delevingne typo
        self.df_results[PERSONA_COL_NAME] = self.df_results[PERSONA_COL_NAME].replace("delevigne", "delevingne")
        
        self.name_of_predicted_column = name_of_predicted_column
        self.add_is_right_column()
        self.df_results.to_csv
    
    def get_results_data_frame(self):
        return self.df_results
    
    def add_is_right_column(self):
        def is_right(predict_name: str, ground_truth_name: str):
            splitted_predicted_name = predict_name.split(" ")
            predicted_family_name = predict_name.lower() if len(splitted_predicted_name) <= 1 else splitted_predicted_name[1].lower()
            return ground_truth_name == predicted_family_name
    
        if PERSONA_COL_NAME not in self.df_results.columns:
            raise Exception("No persona column in the results csv file")
            
        self.df_results["is_right"] = self.df_results.apply(lambda row: is_right(row[self.name_of_predicted_column], row[PERSONA_COL_NAME]), axis=1)


    def persona_accuracy(self, persona: Persona):
        df_persona = self.df_results[persona.value == self.df_results[PERSONA_COL_NAME]]
        if df_persona.shape[0] == 0:
            raise Exception(f"No results for persona family name {persona.value}")
        
        return df_persona["is_right"].mean()

    
