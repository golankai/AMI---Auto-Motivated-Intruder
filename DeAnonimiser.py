import os

from huggingface_hub import notebook_login
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

from utils import get_local_keys, get_prompts_templates, get_repo_id

class DeAnonimiser():
    '''
    Class of a de-anonimiser.
    '''
    def __init__(self, llm: str):
        '''
        Create a new instance of a de-anonimiser.
        :param llm: The LLM to use. Must be one of ['flan-t5' or 'llama2'].
        '''

        # Accesses and keys
        keys = get_local_keys()
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys["huggingface_hub_token"]

        # Define the LLM
        repo_id = get_repo_id(llm)        
        self.llm = llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 512}
        )
        

    def de_anonymise(self, anon_text):
        # Define prompts
        templates = get_prompts_templates()
        prompt = PromptTemplate(template=templates["flanT5"]["pls_de_identify"], input_variables=["anon_text"])

        # Define the chain
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        # Run the chain
        result = llm_chain.run(anon_text)
        return result
    









