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


from utils import get_local_keys, get_prompts_templates, load_model, load_google_search_tool
from PromptBuilder import PromptBuilder


class DeAnonimiser:
    """
    Class of a de-anonimiser.
    """

    def __init__(self, llm: str, self_guide: bool = False, google: bool = False, debug: bool = False, verbose: bool = False):
        """
        Create a new instance of a de-anonimiser.
        :param llm: The LLM to use. Must be one of ['flan-t5' or 'llama2'].
        """

        # Accesses and keys
        langchain.debug = debug
        langchain.verbose = verbose
        self.llm_name = llm
        keys = get_local_keys()
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys["huggingface_hub_token"]

        # Define the PromptBuilder
        self.prompt_builder = PromptBuilder(self.llm_name)

        # Define the LLM
        self.llm = load_model(self.llm_name)

        # Define the ConversationChain
        base_vonv_prompt = self.prompt_builder.get_prompt("base")
        conv_prompt = PromptTemplate(input_variables=["history", "input"], template=self.prompt_builder.get_template("base"))
        self.conversation = ConversationChain(
            prompt=conv_prompt,
            llm=self.llm,
            verbose=True,
            memory=ConversationBufferMemory(verbose=verbose),
        )

        # Define self-guide
        self.self_guide = self_guide
        # Define the google search tool
        self.google = load_google_search_tool() if google else None


    def de_anonymise(self, anon_text):
        answers = {}

        # First try to identify
        prompt, output_parser = self.prompt_builder.get_prompt("pls_de_identify")
        _input = prompt.format_prompt(anon_text=anon_text)
        first_answer = self.conversation.predict(input=_input.to_string())


        # first_answer = output_parser.parse(self.conversation(_input.to_string())["response"])

        print(first_answer)
        print(type(first_answer))
        answers["first_answer"] = first_answer
        # if first_answer != "FAIL":
        #     return answers
        
        # Couldn't identify immidiately, use further methods
        # if self.self_guide:
        #     # Use self-guide
        #     characteristics_cands = self.conversation(self.templates[self.llm_name]["characteristics"])["response"]
        #     answers["characteristics"] = characteristics
        #     cands = self.conversation(self.templates[self.llm_name]["cands"])["response"]

        
        return answers
