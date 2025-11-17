"""
Prompt helpers shared across TiQS training and inference.
"""

PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


def build_prompt(question: str) -> str:
    """
    Format the canonical TiQS prompt for a question string.
    """
    safe_question = question or ""
    return PROMPT_TEMPLATE.format(question=safe_question)
