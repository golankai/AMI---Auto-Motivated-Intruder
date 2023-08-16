import pandas as pd

class DataHandler:
    def __init__(self) -> None:
        self.df = pd.DataFrame()
        self.error_files= pd.DataFrame()
    
    def get_df(self):
        return self.df
    
    def get_error_files(self):
        return self.error_files if self.error_files.shape[0] > 0 else None
    
    def add_flatten_row(self, row, file_name):
        flatten_row = {}
        flatten_row["File"] = file_name

        def flatten_dict(d):
            """
            Flatten a dictionary.
            By separating lists into multiple columns.
            """
            flat_dict = {}
            for key, value in d.items():
                if isinstance(value, list):
                    for idx, item in enumerate(value):
                        new_key = f"{key}_{idx+1}"
                        flat_dict[new_key] = item
                else:
                    flat_dict[key] = value
            return flat_dict

        flatten_row = flatten_dict(row)
        self.add_row(flatten_row)


    def add_row(self, row):
        new_row = pd.DataFrame([row])
        self.df = pd.concat([self.df, new_row], ignore_index=True)


    def add_error_file(self, file_name):
        new_row = pd.DataFrame([{"File": file_name}])
        self.error_files = pd.concat([self.error_files, new_row], ignore_index=True)