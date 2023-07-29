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


from utils import get_local_keys, get_prompts_templates, get_repo_id


class DeAnonimiser:
    """
    Class of a de-anonimiser.
    """

    def __init__(self, llm: str, debug: bool = False, verbose: bool = False):
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

        # Define the templates
        self.templates = get_prompts_templates()

        # Define the LLM
        repo_id = get_repo_id(self.llm_name)
        self.llm = llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 512}
        )

        # Define the ConversationChain
        conv_prompt = PromptTemplate(input_variables=["history", "input"], template=self.templates["conversation"]["base"])
        self.conversation = ConversationChain(
            prompt=conv_prompt,
            llm=self.llm,
            verbose=True,
            # memory=ConversationBufferMemory(return_messages=True)
        )
        self.conversation.memory.chat_memory.add_ai_message(self.templates["conversation"]["ask_for_anon_text"])


    def de_anonymise(self, anon_text):
        # Define prompts
        # prompt = PromptTemplate(
        #     template=self.templates[self.llm_name]["pls_de_identify"],
        #     input_variables=["anon_text"],
        # )
        
        # Define the chain
        # llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        # Run the chain
        # result = llm_chain.run(anon_text)

        # Run the conversation
        result = self.conversation.run(anon_text)
        return result
