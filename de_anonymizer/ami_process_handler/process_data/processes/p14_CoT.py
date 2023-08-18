from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from de_anonymizer.ami_process_handler.process_data.process_data import ProcessData, QueryData

# Study 1, file_id famous_49_d_1_10.txt for the 0 sample
# Study 1, file_id famous_147_d_1_7.txt for the 1 sample

P_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, grading anonymization strength.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P_Q1_TEMPLATE = """
    Task: please rate the anonymization strength of the following anonymized text. Good anonimization reveals few unique information about the person.
    0 is well anonymized and hard to re-identify and 1 is not anonymized at all and can be easily re-identify.
    {format_instructions}

    text: "PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is a NUMERIC_3-year-old, non-binary singer-songwriter from LOCATION_1. They are famous for their distinctive vocals and catchy pop songs, often in collaboration with other pop artists and DJs. For example, they provided the vocals for ORGANIZATION_1's DATE_1 hit single OTHER_IDENTIFYING_ATTRIBUTE_1, which firmly launched PERSON_FIRSTNAME_1's career. Aesthetically, PERSON_FIRSTNAME_1 lost a large amount of weight in recent years, and has since started experimenting more with their look. They have signature dark hair, and often wear jewellery and makeup - particularly eye makeup - and gender non-conforming outfits. They can often be seen performing in skirts and high heels, for example. They are also quite prominent on social media, and regularly post quite intimate updates revealing their innermost feelings about things such as the OTHER_IDENTIFYING_ATTRIBUTE_2 lockdown."
    CoT: "The person is a non-binary singer-songwriter, they have distinctive vocals and catchy pop songs. They  "
    score: 0


    text: "PERSON_FIRSTNAME_2 PERSON_LASTNAME_1 is a very famous actor from the LOCATION_2. PRONOUN is most famous for PRONOUN performances in the OTHER_IDENTIFYING_ATTRIBUTE_4, written by J. K. PERSON_LASTNAME_2, where PRONOUN plays the character of a OTHER_IDENTIFYING_ATTRIBUTE_1. PRONOUN began PRONOUN acting career at a very young age and after finishing the OTHER_IDENTIFYING_ATTRIBUTE_3 films PRONOUN has starred in a number of other roles. PRONOUN began PRONOUN acting career at the age of NUMERIC_4 in a ORGANIZATION_1 television series called OTHER_IDENTIFYING_ATTRIBUTE_6. Since OTHER_IDENTIFYING_ATTRIBUTE_3 PRONOUN has starred in romantic comedies such as OTHER_IDENTIFYING_ATTRIBUTE_5', and thrillers such as OTHER_IDENTIFYING_ATTRIBUTE_2'. PRONOUN has now moved onto producing films and acting on LOCATION_1 where PRONOUN starred in a comedy."
    CoT: ""
    score: 1

    text: "{user_input}"
    score:
"""

class PQ1Parser(BaseModel):
    CoT: str = Field(description="The Chain of Thought")
    score: float = Field(description="The score")

process_14_data = ProcessData(
    base=PromptTemplate(input_variables=["history", "input"], template=P_CONVERSATIONS_BASE_TEMPLATE),
    queries=[
        QueryData(P_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=PQ1Parser)),
    ],
)
