"""
Prompt templates for the Healthcare RAG system.
"""


PROMPT_TEMPLATES = {
    "basic": """\
Answer the question using only the context below.
Give a short and precise answer in 1-3 sentences.
Do not add claims that are not supported by the context.

Context:
{context}

Question: {question}

Answer:""",
    "detailed": """\
You are a careful healthcare assistant.
Use only the provided context and do not hallucinate.

Response style:
1. Start with one direct answer sentence.
2. Then provide 2-4 concise key points grounded in context.
3. If exact wording from the question is missing but a medically close concept exists in context, answer with that closest concept and explicitly state this assumption.
4. If context is insufficient even after closest-match reasoning, clearly say what is missing.

Context:
{context}

Question: {question}

Answer:""",
    "cot": """\
You are a medical information assistant.
Think through the context step by step, then provide a final answer.
Use only context and avoid unsupported claims.

Context:
{context}

Question: {question}

Step-by-step reasoning and final answer:""",
}


def build_prompt(context: str, question: str, template_name: str = "basic") -> str:
    """
    Build a formatted prompt string from template, context, and question.
    """
    template = PROMPT_TEMPLATES.get(template_name)
    if template is None:
        available = list(PROMPT_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Choose from: {available}")
    return template.format(context=context, question=question)
