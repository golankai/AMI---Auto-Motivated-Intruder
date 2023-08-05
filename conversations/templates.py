

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


def get_template(prompt_id: int) -> str:
    """
    Get a template.
    """
    process = int(prompt_id / 10)
    question = prompt_id % 10

    if question == 1:
        return Q1_TEMPLATE
    else:
        raise ValueError("process and question combination is not valid")