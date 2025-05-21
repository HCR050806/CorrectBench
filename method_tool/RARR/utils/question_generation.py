"""Utils for running question generation."""
import os
import time
from typing import List

import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_api_response(api_response: str) -> List[str]:
    """Extract questions from the GPT-3 API response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Question generation response from GPT-3.
    Returns:
        questions: A list of questions.
    """
    search_string = "I googled:"
    questions = []
    for question in api_response.split("\n"):
        # Remove the search string from each question
        if search_string not in question:
            continue
        question = question.split(search_string)[1].strip()
        questions.append(question)

    return questions


def run_rarr_question_generation(
    claim: str,
    model: str,
    prompt: str,
    temperature: float,
    num_rounds: int,
    context: str = None,
    num_retries: int = 1,
) -> List[str]:
    """Generates questions that interrogate the information in a claim.

    Given a piece of text (claim), we use GPT-3 to generate questions that question the
    information in the claim. We run num_rounds of sampling to get a diverse set of questions.

    Args:
        claim: Text to generate questions off of.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        temperature: Temperature to use for sampling questions. 0 represents greedy deconding.
        num_rounds: Number of times to sample questions.
    Returns:
        questions: A list of questions.
    """
    if context:
        gpt3_input = prompt.format(context=context, claim=claim).strip()
    else:
        gpt3_input = prompt.format(claim=claim).strip()

    questions = set()
    for _ in range(num_rounds):
        for _ in range(num_retries):
            try:
                # response = openai.Completion.create(
                #     model=model,
                #     prompt=gpt3_input,
                #     temperature=temperature,
                #     max_tokens=256,
                # )
                print(gpt3_input)
                print("\033[32mQUESTION GENERATION\033[0m")
                response = model.query(gpt3_input)
                print("\033[31m"+response+"\033[0m")
                cur_round_questions = parse_api_response(
                    response.strip()
                )
                questions.update(cur_round_questions)
                break
            except openai.error.OpenAIError as exception:
                print(f"{exception}. Retrying...")
                time.sleep(1)

    questions = list(sorted(questions))
    print("\033[37m")
    print(questions)
    print("\033[0m")
    return questions
