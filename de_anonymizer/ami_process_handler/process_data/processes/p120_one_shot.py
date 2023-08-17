from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from de_anonymizer.ami_process_handler.process_data.process_data import ProcessData, QueryData

# Study 1, file_id famous_147_d_1_7.txt

P_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, grading anonymization strength.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P_Q1_TEMPLATE = """
    Task: please rate the anonymization strength of the following anonymized text.
    {format_instructions}

    text: "PERSON_FIRSTNAME_2 PERSON_LASTNAME_1 is a very famous actor from the LOCATION_2. PRONOUN is most famous for PRONOUN performances in the OTHER_IDENTIFYING_ATTRIBUTE_4, written by J. K. PERSON_LASTNAME_2, where PRONOUN plays the character of a OTHER_IDENTIFYING_ATTRIBUTE_1. PRONOUN began PRONOUN acting career at a very young age and after finishing the OTHER_IDENTIFYING_ATTRIBUTE_3 films PRONOUN has starred in a number of other roles. PRONOUN began PRONOUN acting career at the age of NUMERIC_4 in a ORGANIZATION_1 television series called OTHER_IDENTIFYING_ATTRIBUTE_6. Since OTHER_IDENTIFYING_ATTRIBUTE_3 PRONOUN has starred in romantic comedies such as OTHER_IDENTIFYING_ATTRIBUTE_5', and thrillers such as OTHER_IDENTIFYING_ATTRIBUTE_2'. PRONOUN has now moved onto producing films and acting on LOCATION_1 where PRONOUN starred in a comedy."
    score: 1
    
    text: "{user_input}"
    score:
"""

class PQ1Parser(BaseModel):
    score: float = Field(description="A score from 0 to 1 to evaluate the anonimization, 1 is very easy to re-identify and 0 is impossible")

process_120_data = ProcessData(
    base=PromptTemplate(input_variables=["history", "input"], template=P_CONVERSATIONS_BASE_TEMPLATE),
    queries=[
        QueryData(P_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=PQ1Parser)),
    ],
)
