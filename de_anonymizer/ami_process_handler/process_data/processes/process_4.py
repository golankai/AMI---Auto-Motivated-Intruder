from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from de_anonymizer.ami_process_handler.process_data.process_data import ProcessData, QueryData



P4_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, mimicking an intruder

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P4_Q1_TEMPLATE = """
    We de-identified a specific person in the following anonymized text. Please re-identify this (famous) person based on your knowledge.
    anonymized text: "{user_input}"
    {format_instructions}

    So, the person is
"""

P4_Q2_TEMPLATE = """
    Can you think about another optional person that the anonymized text can be about?
    {format_instructions}
    {user_input}
"""

class P4Q1Parser(BaseModel):
    name_1: str = Field(description="The name of the person based on your knowledge or UNKNOWN if you don't have any idea")

class P4Q2Parser(BaseModel):
    name_2: str = Field(description="Another optional person name based on your knowledge or UNKNOWN if you don't have any idea")


process_4_data = ProcessData(
    base=PromptTemplate(input_variables=["history", "input"], template=P4_CONVERSATIONS_BASE_TEMPLATE),
    queries=[
        QueryData(P4_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P4Q1Parser)),
        QueryData(P4_Q2_TEMPLATE, PydanticOutputParser(pydantic_object=P4Q2Parser)),
    ],
)
