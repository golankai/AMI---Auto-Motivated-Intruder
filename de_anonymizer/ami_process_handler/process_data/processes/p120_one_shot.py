from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from de_anonymizer.ami_process_handler.process_data.process_data import ProcessData, QueryData

# Study 1, file_id famous_49_d_1_10.txt

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

    text: "PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is a NUMERIC_3-year-old, non-binary singer-songwriter from LOCATION_1. They are famous for their distinctive vocals and catchy pop songs, often in collaboration with other pop artists and DJs. For example, they provided the vocals for ORGANIZATION_1's DATE_1 hit single OTHER_IDENTIFYING_ATTRIBUTE_1, which firmly launched PERSON_FIRSTNAME_1's career. Aesthetically, PERSON_FIRSTNAME_1 lost a large amount of weight in recent years, and has since started experimenting more with their look. They have signature dark hair, and often wear jewellery and makeup - particularly eye makeup - and gender non-conforming outfits. They can often be seen performing in skirts and high heels, for example. They are also quite prominent on social media, and regularly post quite intimate updates revealing their innermost feelings about things such as the OTHER_IDENTIFYING_ATTRIBUTE_2 lockdown."
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
