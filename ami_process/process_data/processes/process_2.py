from ami_process.process_data.process_data import ProcessData, QueryData
from typing import List
from pydantic import BaseModel, Field

P2_CONVERSATIONS_BASE_TEMPLATE = """
    bla bla bla
"""

P2_Q1_TEMPLATE = """
    Task: please re-identify the person in the following anonymized text. Note that the anonymized text use placeholders for names, locations, and dates. Your goal is to re-identify the real person and not a placeholder.
    {format_instructions}
    anonymized text: "{user_input}"
"""

P2_RES_COLUMNS = [""]

class P2Q1Parser(BaseModel):
    score: str = Field(description="bla bla bla")

process_2_data = ProcessData (
    base=P2_CONVERSATIONS_BASE_TEMPLATE,
    queries=[
        QueryData(P2_Q1_TEMPLATE, P2Q1Parser),
    ],
    res_columns=P2_RES_COLUMNS
)