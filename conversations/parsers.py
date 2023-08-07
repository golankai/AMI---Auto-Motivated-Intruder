from typing import List
from langchain.output_parsers import StructuredOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, validator

class Q1_parser(BaseModel):
    name: str = Field(description="Name of the person or FAIL")
    score: float = Field(description="Assign a score from 0 to 1 to evaluate the re-identifiabilization, 0 is very easy to re-identify and 1 is impossible")
    characteristics: List[str] = Field(description="A list of 3 unique and identifying characteristics names of the person, short answers")
    # charachteristics_scores: List[str] = Field(description="Assigne a score for the uniquness of a charachteristic, 0 is not unique and 1 is very unique and identifying")

class Q2_parser(BaseModel):
    personas_1: List[str] = Field(description="A list 5 personas names for the first characteristic")
    personas_2: List[str] = Field(description="A list 5 personas names for the second characteristic")
    personas_3: List[str] = Field(description="A list 5 personas names for the third characteristic")

def get_parser(process: int, question: int):
    """
    Get a parser.
    """
    match process, question:
        case 1, 1:
            return PydanticOutputParser(pydantic_object=Q1_parser)
        case 1, 2:
            return PydanticOutputParser(pydantic_object=Q2_parser)
        case _:
            raise ValueError("process and question combination is not valid")