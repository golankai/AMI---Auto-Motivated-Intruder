

CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, mimicking an intruder

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

Q1_TEMPLATE = """Task: please re-identify the person in the following anonymized text. Note that the anonymized text use placeholders for names, locations, and dates. Your goal is to re-identify the real person and not a placeholder.
    {format_instructions}
    anonymized text: "{user_input}"
"""

P1_Q2_TEMPLATE = """
{format_instructions}
Suggest 5 Personas names for each characteristic. (only names)"
{user_input}
"""


def get_template(process: int, question: int) -> str:
    """
    Get a template.
    """
    match process, question:
        case 1, 1:
            return Q1_TEMPLATE
        case 1, 2:
            return P1_Q2_TEMPLATE
        case _:
            raise ValueError("process and question combination is not valid")