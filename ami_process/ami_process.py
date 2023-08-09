from ami_process.parsers import Q1_parser, Q2_parser
from langchain.output_parsers import StructuredOutputParser, PydanticOutputParser

from ami_process.templates import Q1_TEMPLATE, P1_Q2_TEMPLATE


class AMI_process():
    def __init__(self, process) -> None:
        self.process_id = process
        self.question_number = 1

    def new_process(self):
        self.question_number = 1

    def get_res_columns(self): 
        res_columns = {
            1: ["Name", "Score", "Characteristic_1", "Characteristic_2", "Characteristic_3"], # process 1
        }

        return res_columns[self.process_id]

    def __iter__(self):
        return self
    

    def get_parser(self):
        match self.process_id, self.question_number:
            case 1, 1:
                return PydanticOutputParser(pydantic_object=Q1_parser)
            case 1, 2:
                return PydanticOutputParser(pydantic_object=Q2_parser)
            case _:
                # raise ValueError("process and question combination is not valid")
                return None


    def get_template(self):
        match self.process_id, self.question_number:
            case 1, 1:
                return Q1_TEMPLATE
            case 1, 2:
                return P1_Q2_TEMPLATE
            case _:
                # raise ValueError("process and question combination is not valid")
                return None


    def __next__(self):
        template = self.get_template()
        parser = self.get_parser()

        if template is None or parser is None:
            raise StopIteration
        
        self.question_number += 1
        return template, parser
    
