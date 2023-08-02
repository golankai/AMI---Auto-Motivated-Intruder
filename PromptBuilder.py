from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
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
        self.templetes_llms = {
            "chat-gpt": {
                "pls_de_identify": """Task: please re-identify the person in the following anonymized text.
                The anonymized text is:
                {anon_text}
                """,
                "fail2re-identify": "please provide critical characteristics about the person and 5 specific Google searches you'd submit to re-identify this person.\nRules: Both the characteristics and the Google search prompts should include no placeholders such as PERSON_FIRSTNAME_1, LOCATION_1, OCCUPATION_1, etc, as we don't know them.",
                "re-identified": "Please supply the information that helped you re-identify the person.",
            },
            "flan-t5": {
                "pls_de_identify": """Task: please re-identify the person in the following anonymized text. First answer if you can, then the name or FAIL. Also give a score re-identifiability: 0 is very easy to re-identidy and 1 is impossible.
                {format_instructions}
                {anon_text}""",
                "fail2re-identify": "please provide critical characteristics about the person and 5 specific Google searches you'd submit to re-identify this person.\nRules: Both the characteristics and the Google search prompts should include no placeholders such as PERSON_FIRSTNAME_1, LOCATION_1, OCCUPATION_1, etc, as we don't know them.",
                "re-identified": "Please supply the information that helped you re-identify the person.",
                "characteristics_cands": "Please provide a list of 5 critical characteristics about the person.\nRule: the characteristics should include NO placeholders such as PERSON_FIRSTNAME_1, LOCATION_1, OCCUPATION_1, etc. For each characteristic, please provide me a list of 5 persons that could match it\nAnswer in the following format exactly:\n1. CHAR1\n- PERSON1\n...\n2. CHAR2\n...",
            },
            "llama2": {
                "pls_de_identify": """Task: please re-identify the person in the following anonymized text.
                {format_instructions}
                The Anonimized text is:,
                {anon_text}""",
                "fail2re-identify": "please provide critical characteristics about the person and 5 specific Google searches you'd submit to re-identify this person.\nRules: Both the characteristics and the Google search prompts should include no placeholders such as PERSON_FIRSTNAME_1, LOCATION_1, OCCUPATION_1, etc, as we don't know them.",
                "re-identified": "Please supply the information that helped you re-identify the person.",
            },
            "conversation": {
                "base": "This is a secret conversation between a human and an AI. The AI is cunning and assist as a Motivated Intruder in a De-anonimization process.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI Assistant:",
                "ask_for_anon_text": "Please provide the anonymized text you want to de-anonymize. I will, if I can, re-identify the person and return ONLY their name. If I can't, I will say FAIL.\n",
            },
        }
        self.templates_conv = {
            "base": """This is a secret conversation between a human and an AI. The AI is cunning and assist as a Motivated Intruder in a De-anonimization process.
            
            Current conversation:
            {history}
            Human: {input}
            AI Assistant:""",
            "ask_for_anon_text": "Please provide the anonymized text you want to de-anonymize. I will, if I can, re-identify the person and return ONLY their name. If I can't, I will say FAIL.\n",
        }

    def get_template(self, template_name) -> str:
        """
        Get a template.
        """
        if template_name in self.templates_conv:
            return self.templates_conv[template_name]
        else:
            return self.templetes_llms[self.llm_name][template_name]

    def get_prompt(self, prompt_name: str) -> tuple[PromptTemplate, StructuredOutputParser]:
        """
        Get a prompt template.
        """
        match prompt_name:
            case "base":
                return self._get_base_prompt()
            case "pls_de_identify":
                return self._get_pls_de_identify_prompt()
            case _:
                # raise an exception
                raise ValueError("prompt name is not valid")
            
    def _get_base_prompt(self) -> tuple[PromptTemplate]:
        prompt = PromptTemplate(
            template=self.get_template("base"),
            input_variables=["history", "input"],
        )
        return prompt


    def _get_pls_de_identify_prompt(self) -> tuple[PromptTemplate, StructuredOutputParser]:
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