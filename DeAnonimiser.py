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

from utils import get_local_keys, get_prompts_templates

# helper functions

def get_repo_id(llm: str):
    match llm:
        case "flan-t5":
            repo_id = "google/flan-t5-xxl"
        case "llama2":
            repo_id = "meta-llama/Llama-2-70b-hf"
        case _:
            # raise an exception
            raise ValueError("llm name is not valid") 
    return repo_id

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
    



# llm = OpenAI(openai_api_key=keys["openai_api_key"])
# llama = LlamaAPI(keys["llama_api_key"])
# llama_model = ChatLlamaAPI(client=llama)

# Define the chain
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# Run the chain
# anon_text = "PERSON_FIRSTNAME_1 is famous for being an LOCATION_1 songwriter and singer. PRONOUN is from LOCATION_3 in LOCATION_2 and this can be heard heavily in PRONOUN accent even while singing. PERSON_LASTNAME_1!e love PRONOUN music and PRONOUN is famous worldwide. PRONOUN performs live and tours the world with an amazing voice. PRONOUN is loved for PRONOUN unique and powerful voice as well as PRONOUN song writing. PRONOUN songs and albums often about breakups and heartache which many people can relate to. NUMERIC_1 of PRONOUN most famous songs is someone like you. PRONOUN is still only young but has numerous albums that hAve become platinum selling. PRONOUN is now happily married with a child after losing a significant amount of weight which many people were shocked by. PRONOUN is very down to earth and relatable in PRONOUN music"
# result = llm_chain.run(anon_text)
# print(result)
# print(llama_model.predict("Hello, how are you?"))






