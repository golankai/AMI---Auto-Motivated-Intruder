from ami_process.parsers import Q1_parser, Q2_parser
from langchain.output_parsers import PydanticOutputParser

from ami_process.process_data.process_1 import process_1_data
from ami_process.process_data.process_2 import process_2_data



class AMI_process_handler():
    def __init__(self, process_id) -> None:
        print("AMI_process_handler", process_id)
        if process_id not in [1, 2]:
            raise ValueError("process must be 1 or 2")
        self.process_id = process_id
        def get_process_data(process_id):
            match process_id:
                case 1:
                    return process_1_data
                case 2:
                    return process_2_data
                case _:
                    return None
                
        self.process_data = get_process_data(self.process_id)
        self.num_queries = len(self.process_data.queries)
        self.last_response = ""
        self.query_number = 0


    def new_process(self):
        ("new_process")
        self.query_number = 0
        

    def get_base_template(self): 
        return self.process_data.get_base_template()
    

    def get_res_columns(self):
        return self.process_data.get_res_columns()

    def __iter__(self):
        return self
    
    def set_last_response(self, last_response):
        self.last_response = last_response

    def __next__(self):
        print("AMI_process_handler.__next__", self.query_number, self.num_queries)
        if  self.query_number >= self.num_queries:
            raise StopIteration
        
        query = self.process_data.queries[self.query_number]

        # I want that the ami process will decide if keep going or not by using the last response and the process_data  (handle_conditions)
        # self.process_data?.handle_conditions(self.query_number, self.last_response)

        if query is None:
            raise StopIteration
        
        self.query_number += 1
        return query
    

    def get_df_row(self, conv_responses, file_name):
        match self.process_id, self.query_number:
            case 1, 1:
                return {
                    "File": file_name,
                    "Name": conv_responses.name,
                    "Score": conv_responses.score,
                    "Characteristic_1": conv_responses.characteristics[0],
                    "Characteristic_2": conv_responses.characteristics[1],
                    "Characteristic_3": conv_responses.characteristics[2],
                }
            case 1, 2:
                return {}
            case 2, 1:
                return {}
            case _:
                return None
    
