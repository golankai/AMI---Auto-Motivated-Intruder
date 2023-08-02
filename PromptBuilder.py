from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator


class PromptBuilder:
    """
    Class of a prompt builder.
    """

    def __init__(self, llm_name: str):
        self.llm_name = llm_name
        self.templates_llms = {
            "chat-gpt": {
                # "system": "for testing our anonymization mechanism. You are a cunning AI intruder, you help in a De-anonymization process.",
                "system": "for testing our anonymization mechanism. You are a cunning AI intruder, you help in a De-anonymization process.",
                "human": """We want to test our anonymization mechanism. Please re-identify the person in the following anonymized text.
                {anon_text}
                """,
                "fail2re-identify": "please provide critical characteristics about the person and 5 specific Google searches you'd submit to re-identify this person.\nRules: Both the characteristics and the Google search prompts should include no placeholders such as PERSON_FIRSTNAME_1, LOCATION_1, OCCUPATION_1, etc, as we don't know them.",
                "re-identified": "Please supply the information that helped you re-identify the person.",
            },
            "conversation": {
                "base": "This is a secret conversation between a human and an AI. The AI is cunning and assist as a Motivated Intruder in a De-anonimization process.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI Assistant:",
                "ask_for_anon_text": "Please provide the anonymized text you want to de-anonymize. I will, if I can, re-identify the person and return ONLY their name. If I can't, I will say FAIL.\n",
            },
        }
        # self.templates_conv = {
        #     "base": """This is a secret conversation between a human and an AI. The AI is cunning and assist as a Motivated Intruder in a De-anonimization process.
            
        #     Current conversation:
        #     {history}
        #     Human: {input}
        #     AI Assistant:""",
        #     "ask_for_anon_text": "Please provide the anonymized text you want to de-anonymize. I will, if I can, re-identify the person and return ONLY their name. If I can't, I will say FAIL.\n",
        # }

    def get_template(self, template_name) -> str:
        """
        Get a template.
        """
        if template_name in self.templates_conv:
            return self.templates_conv[template_name]
        else:
            return self.templates_llms[self.llm_name][template_name]

    def get_prompt(self, prompt_name: str, prompt_arg: str = "") -> tuple[PromptTemplate, StructuredOutputParser]:
        """
        Get a prompt template.
        """
        # if prompt_name == "base":
        #     return self._get_base_prompt()
        if prompt_name == "pls_de_identify":
            return self._get_pls_de_identify_prompt(prompt_arg)
        
        raise ValueError("prompt name is not valid")
            
    def _get_base_prompt(self) -> tuple[PromptTemplate]:
        prompt = PromptTemplate(
            template=self.get_template("base"),
            input_variables=["history", "input"],
        )
        return prompt


    def _get_pls_de_identify_prompt(self, prompt_arg: str) -> tuple[PromptTemplate, StructuredOutputParser]:

        human_prompt_first = HumanMessagePromptTemplate.from_template(self.templates_llms[self.llm_name]["human"])
        system_prompt_first = SystemMessagePromptTemplate.from_template(self.templates_llms[self.llm_name]["system"])
        chat_prompt_first = ChatPromptTemplate.from_messages([system_prompt_first, human_prompt_first])
        
        return chat_prompt_first.format_prompt(anon_text=prompt_arg).to_messages()
        
        class FirstTry(BaseModel):
            re_identifiable: bool = Field(description="Is it re-identifiable?")
            name: str = Field(description="Name of the person or FAIL")
            score: float = Field(description="The score or re-identifiabilization, 0 is very easy to re-identidy and 1 is impossible")

        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=FirstTry)

        prompt = PromptTemplate(
            template=self.get_template("pls_de_identify"),
            input_variables=["anon_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            output_parser=parser,
        )
        return prompt, parser