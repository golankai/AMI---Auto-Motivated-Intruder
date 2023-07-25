# Imports

# General imports

# LLM imports
from langchain.llms import OpenAI, ChatOpenAI
from langchain.prompts.chat import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

# Local imports

# Define the LLM
llm = OpenAI(openai_api_key="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")


prompt = PromptTemplate.from_template("Please re-identify the following person: {product}")
print(prompt.format(product="ANONIMISEDTEXT"))





print(prompt)
