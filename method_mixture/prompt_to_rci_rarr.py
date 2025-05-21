import re
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from tqdm import tqdm
import argparse
import glob
import sys
import torch
from typing import Dict, Any
import Levenshtein
sys.path.append('/Self-Correction-Benchmark/method_tool/RARR')  

from prompts import hallucination_prompts, rarr_prompts
from utils import (
    agreement_gate,
    editor,
    evidence_selection,
    hallucination,
    search,
    question_generation,
)

class RARR:
    def __init__(self, model, task=None, prompting_style='zero-shot-cot', correct_iteration=1, final_verified_responses=[]):
        self.model = model
        self.task = task
        self.prompting_style = prompting_style
        self.correct_iteration = correct_iteration
        self.final_verified_responses = final_verified_responses
        self.initial_prompt = self.get_initial_prompt()
        # Define any additional prompts or configurations if necessary

    def get_initial_prompt(self):
        if self.prompting_style == 'zero-shot-cot':
            # Here, dynamically include the 'Final Verified Response' from the list
            final_verified_str = '\n'.join(self.final_verified_responses) if self.final_verified_responses else ""
            return (
                "Let's think step by step. \n" + final_verified_str +
                "\nIn the form \\boxed{answer}, at the end of your response.\n\nA:"
            )
        elif self.prompting_style == 'few-shot-cot':
            return "A:\n"  # TODO: add the few-shot prompt file
        elif self.prompting_style == 'zero-shot':
            return (
                "Your final answer in the form \\boxed{answer}, "
                "at the end of your response.\nA:\n"
            )
        else:
            print("WARNING: The prompting style is not given. Use zero-shot-cot as default.")
            return (
                "Let's think step by step.  "
                "in the form \\boxed{answer}, at the end of your response.\nA:\n"
            )

    def get_final_response(self, question, final_answer):
        prompt = f"Q:{question}\nBased on the final conclusion, directly give the answer enclosed in `{{}}` without any further thinking, steps, or explanations: {{}}\n{final_answer}"
        final_response = self.model.query(prompt)
        return final_response
    
    def get_answer(self, output):
        """
        Extracts the answer from the model's output. It looks for the pattern \boxed{answer}.
        """
        
        match = re.findall(r'{(.+?)}', output)
        
        if match:
            answer = match[-1] 
            return answer
        else:
            return None  

    def run_rarr_process(self, claim, context=None):
        """
        Executes the full RARR process: question generation, retrieval, agreement gate, and editing.
        """
        result = run_editor_one_instance(
            claim=claim,
            context=context,
            model=self.model,
            temperature_qgen=0.7,  # These can be parameterized as needed
            num_rounds_qgen=1,
            max_search_results_per_query=5,
            max_sentences_per_passage=4,
            sliding_distance=1,
            max_passages_per_search_result=1,
            max_evidences_per_question=1,
            max_edit_ratio=100,
            hallucinate_evidence=False
        )
        return result
    
    def get_claim(self, question):
        return self.model.query(question)

    def __call__(self, question, answer):
        claim = self.get_claim(question)
            # Print purple divider
        result = self.run_rarr_process(claim, context=None)
        final_answer = self.get_final_answer(result)
        final_respronse = self.get_final_response(question, final_answer)
        final_answer = self.get_answer(final_respronse)
        correct = self.evaluate_answer(final_answer, answer)

        record = {
            "question": question,
            "final_answer": final_answer,
            "correct_answer": answer,
            "correct": correct,
            "result": result
        }

        print("-----------------------------------------")
        print(f"answer:{final_answer}")
        print(f"correct answer:{answer}")
        print("-----------------------------------------")

        if not correct:
            record['error'] = 'Final answer and answer do not match'

        return record

    def get_final_answer(self, result):
        """
        Extracts the final answer from the RARR result.
        """
        selected_evidences = result.get("selected_evidences", [])
        if selected_evidences:
            return selected_evidences[0].get("text", None)
        return None

    def evaluate_answer(self, final_answer, correct_answer):
        """
        Evaluates whether the final answer matches the correct answer.
        """
        if final_answer is None:
            return False
        if str(final_answer) == str(correct_answer):
            return True
        elif str(final_answer) == str(correct_answer)[1:]: 
            return True 
        else:
            return False


def run_editor_one_instance(
    claim: str,
    context: str = None,
    model: str = "text-davinci-003",
    temperature_qgen: float = 0.7,
    num_rounds_qgen: int = 1,
    max_search_results_per_query: int = 5,
    max_sentences_per_passage: int = 4,
    sliding_distance: int = 1,
    max_passages_per_search_result: int = 1,
    max_evidences_per_question: int = 1,
    max_edit_ratio: float = 100,
    hallucinate_evidence: bool = False,
) -> Dict[str, Any]:
    """Runs query generation, search, agreement gating, and editing on a claim.

    Args:
        claim: Text to check the validity of.
        context: Optional context for the claim.
        model: Name of the OpenAI GPT-3 model to use.
        temperature_qgen: Sampling temperature to use for query generation.
        num_rounds_qgen: Number of times to sample questions.
        max_search_results_per_query: Maximum number of search results per query.
        max_sentences_per_passage: Maximum number of sentences for each passage.
        sliding_distance: Sliding window distance over the sentences of each search
            result. Used to extract passages.
        max_passages_per_search_result:  Maximum number of passages to return for
            each search result. A passage ranker is applied first.
        max_evidences_per_question: Maximum number of evidences to return per question.
        max_edit_ratio: Maximum edit ratio between claim and edit for each round.
        hallucinate_evidence: Whether to hallucinate evidence instead of retrieving it.
    Returns:
        result: All revision information, including the queries generated, search
            results, agreement gate information, and each revision step done on the
            claim.
    """
    original_claim = claim
    agreement_gates = []
    questions = question_generation.run_rarr_question_generation(
        claim=claim,
        context=context,
        model=model,
        prompt=rarr_prompts.CONTEXTUAL_QGEN_PROMPT
        if context
        else rarr_prompts.QGEN_PROMPT,
        temperature=temperature_qgen,
        num_rounds=num_rounds_qgen,
    )
    if hallucinate_evidence:
        raise_hallucinate_evidence_warning()
        evidences_for_questions = [
            [
                hallucination.run_evidence_hallucination(
                    query=query,
                    model=model,
                    prompt=hallucination_prompts.EVIDENCE_HALLUCINATION,
                )
            ]
            for query in questions
        ]
    else:
        evidences_for_questions = [
            search.run_search(
                query=query,
                max_search_results_per_query=max_search_results_per_query,
                max_sentences_per_passage=max_sentences_per_passage,
                sliding_distance=sliding_distance,
                max_passages_per_search_result_to_return=max_passages_per_search_result,
            )
            for query in questions
        ]
    used_evidences = [
        e
        for cur_evids in evidences_for_questions
        for e in cur_evids[:max_evidences_per_question]
    ]
    revision_steps = []
    for evid in used_evidences:
        gate = agreement_gate.run_agreement_gate(
            claim=claim,
            context=context,
            query=evid["query"],
            evidence=evid["text"],
            model=model,
            prompt=rarr_prompts.CONTEXTUAL_AGREEMENT_GATE_PROMPT
            if context
            else rarr_prompts.AGREEMENT_GATE_PROMPT,
        )
        agreement_gates.append(gate)

        if gate["is_open"]:
            edited_claim = editor.run_rarr_editor(
                claim=claim,
                context=context,
                query=evid["query"],
                evidence=evid["text"],
                model=model,
                prompt=rarr_prompts.CONTEXTUAL_EDITOR_PROMPT
                if context
                else rarr_prompts.EDITOR_PROMPT,
            )["text"]

            if Levenshtein.distance(claim, edited_claim) / len(claim) <= max_edit_ratio:
                claim = edited_claim

        revision_steps.append({"text": claim})
    revised_text = revision_steps[-1]["text"] if revision_steps else "Default revision text"
    result = {
        "context": context,
        "text": original_claim,
        "questions": questions,
        "evidences_for_questions": evidences_for_questions,
        "revisions": [
            {
                "original_text": original_claim,
                "revised_text": revised_text,
                "evidences": used_evidences,
                "agreement_gates": agreement_gates,
                "revision_steps": revision_steps,
            }
        ],
    }
    selected_evidences = evidence_selection.select_evidences(result)
    result["selected_evidences"] = selected_evidences
    return result

def raise_hallucinate_evidence_warning():
    if not raise_hallucinate_evidence_warning.called:
        print(
            "WARNING!! YOU ARE USING A LLM TO GENERATE EVIDENCE POTENTIALLY WITH "
            "HALLUCINATIONS INSTEAD OF RETRIEVING EVIDENCE. \n\nThis should NEVER be "
            "done when trying to improve attribution as evidence may be inaccurate "
            "and is only provided to quickly experiment with repository setting up "
            "the search API first.\n"
        )
    raise_hallucinate_evidence_warning.called = True

raise_hallucinate_evidence_warning.called = False

import os
import json
import argparse
import sys
from tqdm import tqdm
import torch


def read_final_verified_responses(file_path):
    """读取JSON文件并提取final_verified_responses"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    final_verified_responses = []
    
    if 'results' in data and isinstance(data['results'], list):
        for item in data['results']:
            response = item.get('round_0')
            if response:
                final_verified_responses.append(response)
            else:

                final_verified_responses.append(None)  
    else:
        print("JSON 数据中没有 'results' 键或 'results' 不是一个列表。")
    print("提取到的数量为：", len(final_verified_responses))
    return final_verified_responses

def test_and_save(args):
    sys.path.append('/Self-Correction-Benchmark')  
    from utils.process_config import open_config
    sys.path.append('/Self-Correction-Benchmark/method_tool/RARR')  
    from model import create_model
    from task import create_task
    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    task_config_path = args.task_config

    if not os.path.exists(task_config_path):
        print(f"Task configuration file not found at {task_config_path}.")
        return

    task_config = open_config(config_path=task_config_path)
    task = create_task(task_config)
    data = task.get_data()

    # 读取final_verified_responses
    final_verified_responses = read_final_verified_responses(args.final_verified_responses_file)

    method = RARR(model, task, args.prompting_style, correct_iteration=args.correct_iteration, final_verified_responses=final_verified_responses)

    results_path = f'/Self-Correction-Benchmark/results_tool/{args.method}/{task.task_name}/'
    results_file = f'{results_path}/{model.name}_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    final_results = []
    correct_number = 0
    total_number = len(data['question'])
    empty_answer_count = 0

    for i in tqdm(range(total_number)):
        question = data['question'][i]
        answer = data['final_answer'][i]
        record = method(question, answer)
        final_results.append(record)
        
        if record and record.get('correct', False):
            correct_number += 1
        if record and record.get('final_answer') is None:
            empty_answer_count += 1

        ACC = correct_number / (i + 1 - empty_answer_count) if (i + 1 - empty_answer_count) > 0 else 0
        results_dict = {
            "ACC": ACC,
            "empty_answers": empty_answer_count,
            "results": final_results
        }

        with open(results_file, 'w', encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4)

    print(f"Method: {args.method}\nTask: {task.task_name}\nModel: {model.name}\nFinal Accuracy: {ACC:.2f}")
    print(f"Number of questions with empty answers: {empty_answer_count}")
    print(f"Results saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description="RARR Testing and Saving Script for a Single Task")
    parser.add_argument('--model_config', type=str, default='/Self-Correction-Benchmark/config/model_config/api_LLaMA3.1-70B.json',
                        help='Path to the model configuration file.')
    parser.add_argument('--task_config', type=str, default='/Self-Correction-Benchmark/config/task_config_all/gpqa.json',
                        help='Path to the task configuration file.')
    parser.add_argument('--method', type=str, default='prompt_to_rci_rarr',
                        help='Method name to use.')
    parser.add_argument('--prompting_style', type=str, default='zero-shot-cot',
                        choices=['zero-shot-cot', 'few-shot-cot', 'zero-shot'],
                        help='Prompting style to use.')
    parser.add_argument('--correct_iteration', type=int, default=1, 
                        help='Number of correction iterations.')
    parser.add_argument('--final_verified_responses_file', type=str, 
                        default='/Self-Correction-Benchmark/results/prompt_to_rci/GPQA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo_results.json',
                        help='Path to the JSON file containing final_verified_responses.')
    args = parser.parse_args()

    test_and_save(args)

if __name__ == "__main__":
    main()
