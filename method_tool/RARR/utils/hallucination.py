"""Utils for generating fake evidence given a query."""
import os
import time
from typing import Dict

import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")


def run_evidence_hallucination(
    query: str,
    model: str,
    prompt: str,
    num_retries: int = 1,
) -> Dict[str, str]:
    """Generates a fake piece of evidence via LLM given the question.

    Args:
        query: Query to guide the validity check.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        output: A potentially inaccurate piece of evidence.
    """
    gpt3_input = prompt.format(query=query).strip()
    for _ in range(num_retries):
        try:
        #     stop=["\n", "\n\n"]
        #     for s in stop:
        # # 查找停止符的位置
        #         index = text.find(s)
        #         if index != -1:  # 如果找到停止符
        #             return text[:index]  # 截取停止符之前的部分
            # response = openai.Completion.create(
            #     model=model,
            #     prompt=gpt3_input,
            #     temperature=0.0,
            #     max_tokens=256,
            #     stop=["\n", "\n\n"],
            # )
            print(gpt3_input)
            print("\033[32mHALLUCINATION\033[0m")
            response = model.query(gpt3_input)
            print("\033[31m"+response+"\033[0m")
            stop=["\n", "\n\n"]
            for s in stop:
        # 查找停止符的位置
                index = response.find(s)
                if index != -1:  # 如果找到停止符
                    response = response[:index]  # 截取停止符之前的部分
                    break
        except openai.error.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(2)

    hallucinated_evidence = response.choices[0].text.strip()
    output = {"text": hallucinated_evidence, "query": query}
    return output
