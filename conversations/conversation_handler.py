import os
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field, validator
from conversations.parsers import get_parser

from conversations.templates import CONVERSATIONS_BASE_TEMPLATE, get_template


class ConversationHandler:
    def __init__(self, llm_chat_model):
        self.process_number = -1
        self.question_number = -1

        self.conversation = None
        
        self.llm_chat_model = llm_chat_model
        self.conv_base_prompt = PromptTemplate(input_variables=["history", "input"], template=CONVERSATIONS_BASE_TEMPLATE)
    
    def start_conversation(self, process_number: int):
        # Code to start a new conversation with OpenAI API
        self.process_number = process_number
        self.question_number = 1
        self.conversation = ConversationChain(
            llm=self.llm_chat_model,
            memory=ConversationBufferMemory(return_messages=True),
            prompt=self.conv_base_prompt
        )

    def new_human_prompt(self, user_input: str):
        # Code to add a prompt to the current conversation
        # prompt_id is defined according to the process and the question numbers
        # e.g. "11" for the first question of the first process
        template = get_template(self.process_number, self.question_number)
        parser = get_parser(self.process_number, self.question_number)
        human_prompt = HumanMessagePromptTemplate.from_template(template=template, output_parser=parser)
        human_prompt = human_prompt.format(user_input=user_input, format_instructions=parser.get_format_instructions())
        return human_prompt, parser

    def send_new_message(self, user_input: str=""):
        human_prompt, parser = self.new_human_prompt(user_input)
        # Code to send user input and get a response from OpenAI API
        response = self.conversation.predict(input=human_prompt.content)

        self.question_number += 1
        return parser.parse(response)

    def end_conversation(self):
        # Code to end the current conversation
        self.conversation.memory.clear()
