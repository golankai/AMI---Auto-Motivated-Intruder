from de_anonymizer.processes.p1_gk_one import process_1_data
from de_anonymizer.processes.p2_guess3 import process_2_data
from de_anonymizer.processes.p3_complete_sent import process_3_data
from de_anonymizer.processes.p4_gk_multi import process_4_data
from de_anonymizer.processes.p5_zero_shot_conf_guess import process_5_data
from de_anonymizer.processes.p5_zero_shot import process_5_1_data
from de_anonymizer.processes.p5_2_goal_wo_knowledge import process_5_2_data

# Grader imports
from anon_grader.processes.p11_zero_shot_grader import process_11_data
from anon_grader.processes.p111_multi_persona import process_111_data
from anon_grader.processes.p121_one_shot import process_121_data
from anon_grader.processes.p120_one_shot import process_120_data
from anon_grader.processes.p13_three_shot import process_13_data
from anon_grader.processes.p14_CoT import process_14_data
from anon_grader.processes.p161_role1 import process_161_data
from anon_grader.processes.p162_role2 import process_162_data
from anon_grader.processes.p163_role3 import process_163_data
from anon_grader.processes.p164_role4 import process_164_data


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