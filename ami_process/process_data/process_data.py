from dataclasses import dataclass
from typing import Optional
from enum import Enum

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    PromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

class MessageType(Enum):
    HUMAN = 0
    SYSTEM = 1

@dataclass
class QueryData:
    template: str
    parser: PydanticOutputParser
    type: Optional[MessageType] = MessageType.HUMAN

    def get_prompt(self) -> HumanMessagePromptTemplate | SystemMessagePromptTemplate:
        template, parser = self.template, self.parser
        prompt = None
        if self.type == MessageType.HUMAN:
            prompt = HumanMessagePromptTemplate.from_template(template=template, output_parser=parser)
        if self.type == MessageType.SYSTEM:
            prompt = SystemMessagePromptTemplate.from_template(template=template, output_parser=parser)
        
        return prompt


@dataclass
class ProcessData:
    base: PromptTemplate
    queries: list[QueryData]
    res_columns: list[str]

    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, index) -> QueryData:
        if index >= len(self.queries):
            raise StopIteration
        else:
            return self.queries[index]
        
    def get_res_columns(self) -> list[str]:
        return self.res_columns
    
    def get_base_template(self) -> PromptTemplate:
        return self.base
    
    def get_row(self, response):
        return {col: response[col] for col in self.get_res_columns()}
