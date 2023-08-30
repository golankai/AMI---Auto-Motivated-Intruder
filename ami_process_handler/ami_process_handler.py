from .process_data.processes.p1_direct import process_1_data
from .process_data.processes.p2_guess3 import process_2_data
from .process_data.processes.p3_complete_sent import process_3_data
from .process_data.processes.p4_LoMP import process_4_data
from .process_data.processes.p5_guess1 import process_5_data
from .process_data.processes.p5_1_no_goal import process_5_1_data
from .process_data.processes.p5_2_goal_wo_knowledge import process_5_2_data

# Grader imports
from .process_data.processes.p11_zero_shot_grader import process_11_data
from .process_data.processes.p111_multi_persona import process_111_data
from .process_data.processes.p121_one_shot import process_121_data
from .process_data.processes.p120_one_shot import process_120_data
from .process_data.processes.p13_three_shot import process_13_data
from .process_data.processes.p14_CoT import process_14_data
from .process_data.processes.p161_role1 import process_161_data
from .process_data.processes.p162_role2 import process_162_data
from .process_data.processes.p163_role3 import process_163_data
from .process_data.processes.p164_role4 import process_164_data

class AMI_process_handler:
    def __init__(self, process_id) -> None:
        self.process_id = process_id

        def get_process_data(process_id):
            match process_id:
                case 1:
                    return process_1_data
                case 2:
                    return process_2_data
                case 3:
                    return process_3_data
                case 4:
                    return process_4_data
                case 5:
                    return process_5_data
                case 51:
                    return process_5_1_data
                case 52:
                    return process_5_2_data
                case 11:
                    return process_11_data
                case 120:
                    return process_120_data
                case 121:
                    return process_121_data
                case 13:
                    return process_13_data
                case 14:
                    return process_14_data
                case 111:
                    return process_111_data
                case 161:
                    return process_161_data
                case 162:
                    return process_162_data
                case 163:
                    return process_163_data
                case 164:
                    return process_164_data
                case _:
                    raise ValueError("you must match your process data with the id.")

        self.process_data = get_process_data(self.process_id)
        self.num_queries = len(self.process_data.queries)
        self.query_number = 0
        self.conv_responses = {}

    def new_process(self):
        self.query_number = 0
        self.conv_responses = {}

    def get_base_template(self):
        return self.process_data.get_base_template()

    def get_conv_responses(self):
        return self.conv_responses

    def __iter__(self):
        return self

    def set_last_response(self, last_response):
        self.conv_responses.update(last_response.dict())

    def __next__(self):
        if self.query_number >= self.num_queries:
            raise StopIteration

        query = self.process_data.queries[self.query_number]

        # I want that the ami process will decide if keep going or not by using the last response and the process_data  (handle_conditions)
        # self.process_data?.handle_conditions(self.query_number, self.last_response)

        if query is None:
            raise StopIteration

        self.query_number += 1
        return query
