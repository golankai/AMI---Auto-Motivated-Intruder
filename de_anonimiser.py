# Imports

# General imports

# LLM imports
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

# Local imports
from utils import get_local_keys

# Get local keys
keys = get_local_keys()

# Define prompts
prompt = PromptTemplate.from_template("Please re-identify the following person: {product}")

# Define the LLM
llm = OpenAI(openai_api_key=keys["openai_api_key"])
llama = LlamaAPI(keys["llama_api_key"])
llama_model = ChatLlamaAPI(client=llama)


# Define the chain

# Run the chain
print(llama_model.predict("Hello, how are you?"))





