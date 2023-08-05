import os

from huggingface_hub import notebook_login
import langchain
from langchain import HuggingFacePipeline
from langchain.llms import OpenAI, HuggingFaceHub, Cohere
from langchain.prompts.chat import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from conversations.conversation_handler import ConversationHandler

from utils import get_local_keys, load_google_search_tool, load_model


class DeAnonymizer:
    """
    Class of a de-anonimiser.
    """

    def __init__(self, llm_name: str, self_guide: bool = False, google: bool = False, debug: bool = False, verbose: bool = False):
        """
        Create a new instance of a de-anonymiser.
        :param llm: The LLM to use.
        """

        # Accesses and keys
        langchain.debug = debug
        langchain.verbose = verbose
        self.llm_name = llm_name
        keys = get_local_keys()
        # os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys["huggingface_hub_token"]
        os.environ["OPENAI_API_KEY"] = keys["openai_api_key"]

        # Define the LLM and the conversation handler
        self.llm = load_model(self.llm_name)
        self.conversation_handler = ConversationHandler(self.llm)

        # Define self-guide
        self.self_guide = self_guide
        # Define the google search tool
        self.google = load_google_search_tool() if google else None


    def de_anonymise(self, anon_text):
        self.conversation_handler.start_conversation()
        response = self.conversation_handler.send_new_message(prompt_id=11, user_input=anon_text)

        print(response)
        # first_answer = self.llm(prompt)


        # first_answer = output_parser.pred(self.conversation(_input.to_string())["response"])

        # print(first_answer)
        # print(type(first_answer))
        # answers["first_answer"] = first_answer
        # if first_answer != "FAIL":
        #     return answers
        
        # Couldn't identify immidiately, use further methods
        # if self.self_guide:
        #     # Use self-guide
        #     characteristics_cands = self.conversation(self.templates[self.llm_name]["characteristics"])["response"]
        #     answers["characteristics"] = characteristics
        #     cands = self.conversation(self.templates[self.llm_name]["cands"])["response"]

        self.conversation_handler.end_conversation()

        return response
