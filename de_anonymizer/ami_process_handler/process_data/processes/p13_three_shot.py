from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from de_anonymizer.ami_process_handler.process_data.process_data import ProcessData, QueryData

# Study 1, file_id famous_49_d_1_10.txt for the 0 sample
# Study 1, file_id famous_200_d_1_4.txt for the 0.5 sample 
# Study 1, file_id famous_147_d_1_7.txt for the 1 sample

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
    score: 0

    text: 'PERSON_FIRSTNAME_2 is a NUMERIC_2 year old singer and composer from LOCATION_1. Early in PRONOUN life PRONOUN enjoyed playing piano by ear, and fancied a career in music. PRONOUN first job was as a pub pianist, where PRONOUN played well known tunes as well as PRONOUN own original songs. As an adult, PERSON_FIRSTNAME_2 wrote many famous songs such as Candle in the Wind, Can You Feel The Love Tonight and OTHER_IDENTIFYING_ATTRIBUTE_1. PRONOUN composed the music for OTHER_IDENTIFYING_ATTRIBUTE_3 PERSON_LASTNAME_5 the OTHER_IDENTIFYING_ATTRIBUTE_4. PERSON_FIRSTNAME_2 was originally engaged to be married to PERSON_FIRSTNAME_4 PERSON_LASTNAME_3, but the relationship ended before the marriage. PERSON_FIRSTNAME_2 eventually married PERSON_FIRSTNAME_3 PERSON_LASTNAME_2 in DATE_3, and was divorced in DATE_7. PERSON_FIRSTNAME_2 has come out twice, first as bisexual in DATE_4, and then as gay in DATE_2, both times via the rolling stone magazine. PRONOUN begun a relationship with PERSON_LASTNAME_1 PERSON_LASTNAME_4 in DATE_6, with whom PRONOUN had a civil partnership in DATE_5, and married in DATE_1. PERSON_FIRSTNAME_2 still tours, however in recent years PRONOUN has hinted at stepping back to spend more time with PRONOUN family.'
    score: 0.5

    text: "PERSON_FIRSTNAME_2 PERSON_LASTNAME_1 is a very famous actor from the LOCATION_2. PRONOUN is most famous for PRONOUN performances in the OTHER_IDENTIFYING_ATTRIBUTE_4, written by J. K. PERSON_LASTNAME_2, where PRONOUN plays the character of a OTHER_IDENTIFYING_ATTRIBUTE_1. PRONOUN began PRONOUN acting career at a very young age and after finishing the OTHER_IDENTIFYING_ATTRIBUTE_3 films PRONOUN has starred in a number of other roles. PRONOUN began PRONOUN acting career at the age of NUMERIC_4 in a ORGANIZATION_1 television series called OTHER_IDENTIFYING_ATTRIBUTE_6. Since OTHER_IDENTIFYING_ATTRIBUTE_3 PRONOUN has starred in romantic comedies such as OTHER_IDENTIFYING_ATTRIBUTE_5', and thrillers such as OTHER_IDENTIFYING_ATTRIBUTE_2'. PRONOUN has now moved onto producing films and acting on LOCATION_1 where PRONOUN starred in a comedy."
    score: 1

    text: "{user_input}"
    score:
"""

class P12Q1Parser(BaseModel):
    score: float = Field(description="A score from 0 to 1 to evaluate the anonimization, 1 is very easy to re-identify and 0 is impossible")

process_13_data = ProcessData(
    base=PromptTemplate(input_variables=["history", "input"], template=P_CONVERSATIONS_BASE_TEMPLATE),
    queries=[
        QueryData(P_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P12Q1Parser)),
    ],
)
