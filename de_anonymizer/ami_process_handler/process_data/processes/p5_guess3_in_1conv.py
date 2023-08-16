from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from de_anonymizer.ami_process_handler.process_data.process_data import ProcessData, QueryData



P5_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. 
    The AI is helping the user validate their anonymization process, mimicking an intruder.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P5_Q1_TEMPLATE = """
    Task: please re-identify the person in the following anonymized text.
    Note that the anonymized text use placeholders. 
    Your goal is to guess the famous person based on the anonymized text and by using your knowledge.
    
    {format_instructions}
    
    anonymized text: "{user_input}"
"""

P5_Q23_TEMPLATE = """
    Are you sure ? you can change your answer if you want to.
    
    {format_instructions}
    {user_input}
"""

class P5Q123Parser(BaseModel):
    name: str = Field(description="Name of the person or FAIL")
    score: float = Field(description="Assign a score from 0 to 1 to evaluate the re-identifiabilization, 0 is very easy to re-identify and 1 is impossible")
    characteristics: List[str] = Field(description="A list of 3 unique and identifying characteristics names of the person, short answers")



process_5_data = ProcessData(
    base=PromptTemplate(input_variables=["history", "input"], template=P5_CONVERSATIONS_BASE_TEMPLATE),
    queries=[
        QueryData(P5_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P5Q123Parser)),
        QueryData(P5_Q23_TEMPLATE, PydanticOutputParser(pydantic_object=P5Q123Parser)),
        QueryData(P5_Q23_TEMPLATE, PydanticOutputParser(pydantic_object=P5Q123Parser))
    ],
)
