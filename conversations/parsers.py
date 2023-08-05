from langchain.output_parsers import StructuredOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, validator

class Q1_parser(BaseModel):
    name: str = Field(description="Name of the person or FAIL")
    score: float = Field(description="Assign a score from 0 to 1 to evaluate the re-identifiabilization, 0 is very easy to re-identify and 1 is impossible")
    # charachteristics: List[str] = Field(description="A list of 3 unique and identifying charachteristics names of the person, short answers")


def get_parser(promt_id: int):
    """
    Get a parser.
    """
    process = int(promt_id / 10)
    question = promt_id % 10
    
    if question == 1:
        return PydanticOutputParser(pydantic_object=Q1_parser)
    else:
        raise ValueError("process and question combination is not valid")