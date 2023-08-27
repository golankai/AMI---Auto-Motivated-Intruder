import pandas as pd


class DataHandler:
    def __init__(self) -> None:
        self.df = pd.DataFrame()
        self.error_files = pd.DataFrame()

    def get_df(self):
        return self.df

    def get_error_files(self):
        return self.error_files if self.error_files.shape[0] > 0 else None

    def add_flatten_row(self, row, file_name):
        flatten_row = {}

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
        flatten_row["File"] = file_name

        self.add_row(flatten_row)

    def add_row(self, row):
        new_row = pd.DataFrame([row])
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def add_error_file(self, file_name, raw_response):
        new_row = pd.DataFrame([{"File": file_name, "Raw_response": raw_response}])
        self.error_files = pd.concat([self.error_files, new_row], ignore_index=True)

    def save_to_csv(self, path, error_files_path):
        if self.error_files.shape[0] > 0:
            self.error_files.to_csv(error_files_path, index=False)
            print("Save error files to csv successfully! file-name: ", error_files_path)

        self.df.to_csv(path, index=False)
        print("Save to csv successfully! file-name: ", path)
