import json
import os
import pandas as pd
from results.paths import ResultsPaths
from personas.personas import Persona
from enum import Enum

"""
    EXPERIMENT EVALUATION
    ---------------------

    This class is used to evaluate the results of the experiment.
    It can be used to process the raw results and save them to a csv file.
    Also, it can be used to evaluate the results and save them to a json file.

    Assumption:
    1. The raw results columns includes the following columns:
        - File: the file name
        - name: the name of the person
"""

class ColNames(Enum):
    PERSONA = "Persona"
    REIDENTIFICATION = "is_reidentified"
    FILE = "File"
    NAME = "name"

class ExperimentEvaluation:
    def __init__(self, results_paths: ResultsPaths):
        self.evaluation: dict = {}
        self.results_paths = results_paths

        if os.path.exists(self.results_paths.processed):
            self.df_results = pd.read_csv(self.results_paths.processed)
        else:
            self.df_results = pd.read_csv(self.results_paths.raw)

            if ColNames.NAME.value not in self.df_results.columns or ColNames.FILE.value not in self.df_results.columns:
                raise Exception("The raw results file should include the following columns: File, name. current columns: ", self.df_results.columns)
            
            # Add persona column
            self.add_persona_column()
            self.process_results()
        
        
    def process_results(self):
        # Sort by file name
        self.df_results.sort_values(by=[ColNames.FILE.value], inplace=True)

        # Add is_right column
        self.add_is_re_identify_successfully_column()

        # Save results
        self.df_results.to_csv(self.results_paths.processed, index=False)


    def persona_successful_rate(self):
        for persona in Persona:
            persona_df = self.df_results[self.df_results[ColNames.PERSONA.value] == persona.value]
            num_of_files = persona_df.shape[0]
            if num_of_files <= 0:
                continue

            num_of_re_identify_successfully = int(persona_df[ColNames.REIDENTIFICATION.value].sum()) # convert int64 to native int (for json)

            self.evaluation[persona.value] = {
                "successful_identification_rate": num_of_re_identify_successfully / num_of_files,
                "num_of_files": num_of_files,
                "num_of_re_identify_successfully": num_of_re_identify_successfully,
            }


    def overall_successful_rate(self):
        num_of_files = self.df_results.shape[0]
        num_of_re_identify_successfully = int(self.df_results[ColNames.REIDENTIFICATION.value].sum()) # convert int64 to native int (for json)
        
        self.evaluation["overall"] = {
            "successful_identification_rate": num_of_re_identify_successfully / num_of_files,
            "num_of_files": num_of_files,
            "num_of_re_identify_successfully": num_of_re_identify_successfully,
        }

    
    def add_persona_column(self):
        def get_persona_value_by_file(file_name: str):
            # Inputs:
            # :file_name: str - in the format of: <persona_family_name>_<text_number>.txt
            # We still using the enum to get error if something unexpected happens

            family_name = file_name.split("_")[0].capitalize()
            return Persona[family_name].value
        
        self.df_results[ColNames.PERSONA.value] = self.df_results.apply(lambda row: get_persona_value_by_file(row[ColNames.FILE.value]), axis=1)


    def add_is_re_identify_successfully_column(self):
        def is_re_identify_successfully(predicted_name: str, persona_value: str):
            persona = Persona[persona_value.capitalize()]
            return predicted_name in persona.get_optional_names()

        self.df_results[ColNames.REIDENTIFICATION.value] = self.df_results.apply(lambda row: is_re_identify_successfully(row[ColNames.NAME.value], row[ColNames.PERSONA.value]), axis=1)
    

    def get_results_data_frame(self):
        return self.df_results
    

    def to_json(self):
        path = self.results_paths.evaluation
        if os.path.exists(path):
            with open(path, "r") as f:
                json_data = json.load(f)

            json_data.update(self.evaluation)
            self.evaluation = json_data

        with open(path, "w") as f:
            json.dump(self.evaluation, f, indent=4)


    
