from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)


class PromptBuilder:
    """
    Class of a prompt builder.
    """

    def __init__(self, llm_name: str):
        self.llm_name = llm_name
        self.templetes_llms = {
            "chat-gpt": {
                "pls_de_identify": """Task: please re-identify the person in the following anonymized text. If you can, answer 'Person: name'. Else say No.
            {anon_text}""",
                "fail2re-identify": "please provide critical characteristics about the person and 5 specific Google searches you'd submit to re-identify this person.\nRules: Both the characteristics and the Google search prompts should include no placeholders such as PERSON_FIRSTNAME_1, LOCATION_1, OCCUPATION_1, etc, as we don't know them.",
                "re-identified": "Please supply the information that helped you re-identify the person.",
            },
            "flan-t5": {
                "pls_de_identify": """Task: please re-identify the person in the following anonymized text.
                {format_instructions}
                {anon_text}""",
                "fail2re-identify": "please provide critical characteristics about the person and 5 specific Google searches you'd submit to re-identify this person.\nRules: Both the characteristics and the Google search prompts should include no placeholders such as PERSON_FIRSTNAME_1, LOCATION_1, OCCUPATION_1, etc, as we don't know them.",
                "re-identified": "Please supply the information that helped you re-identify the person.",
                "characteristics_cands": "Please provide a list of 5 critical characteristics about the person.\nRule: the characteristics should include NO placeholders such as PERSON_FIRSTNAME_1, LOCATION_1, OCCUPATION_1, etc. For each characteristic, please provide me a list of 5 persons that could match it\nAnswer in the following format exactly:\n1. CHAR1\n- PERSON1\n...\n2. CHAR2\n...",
            },
            "llama2": {
                "pls_de_identify": "Task: please re-identify the person in the following anonymized text. If you can, answer 'Person: name'. Else say No.\n {anon_text}",
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
            case "pls_de_identify":
                return self._get_pls_de_identify_prompt()
            case "llama2":
                repo_id = "meta-llama/Llama-2-70b-hf"
            case _:
                # raise an exception
                raise ValueError("prompt name is not valid")
            

    def _get_pls_de_identify_prompt(self) -> tuple[PromptTemplate, StructuredOutputParser]:
        response_schemas = [
        ResponseSchema(name="first_answer", description="Name of the person or FAIL", type="str"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=self.get_template("pls_de_identify"),
            input_variables=["anon_text"],
            partial_variables={"format_instructions": format_instructions},
            output_parser=output_parser,
        )
        return prompt, output_parser