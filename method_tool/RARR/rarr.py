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
sys.path.append('/Self-Correction-Benchmark/method_tool/RARR')  # 请根据实际路径修改

# Ensure CUDA_VISIBLE_DEVICES is set
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 根据实际情况设置
# print("CUDA是否可用:", torch.cuda.is_available())
# print("可用的GPU数量:", torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

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
    def __init__(self, model, task=None, prompting_style='zero-shot-cot', correct_iteration=1):
        self.model = model
        self.task = task
        self.prompting_style = prompting_style
        self.correct_iteration = correct_iteration
        self.initial_prompt = self.get_initial_prompt()
        # Define any additional prompts or configurations if necessary

    def get_initial_prompt(self):
        if self.prompting_style == 'zero-shot-cot':
            return (
                "Let's think step by step. "
                "In the form \\boxed{answer}, at the end of your response.\n\nA:"
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
        # print(prompt)
        final_response = self.model.query(prompt)
        return final_response
    def get_answer(self, output):
        """
        Extracts the answer from the model's output. It looks for the pattern \boxed{answer}.
        """
        
        match = re.findall(r'{(.+?)}', output)
        
        if match:
            # try:
            #     return int(answer[0])
            # except ValueError:
            #     try:
            #         return float(answer[0])
            #     except ValueError:
            #         return answer[0] 
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
        # try:
            # context = self.task.get_context(question) if self.task else None  # Adjust based on task
            # context = self.get_context(question)
            claim = self.get_claim(question)
            # 打印紫色分割线
            # print("\033[35m" + "=" * 50 + "\033[0m")
            result = self.run_rarr_process(claim, context=None)
            # 打印紫色分割线
            # print("\033[35m" + "=" * 50 + "\033[0m")
            final_answer = self.get_final_answer(result)
            final_respronse = self.get_final_response(question, final_answer)
            # print("\033[32m" + final_respronse + "\033[0m")
            final_answer = self.get_answer(final_respronse)
            # print("\033[32m" + final_answer + "\033[0m")
            # 打印紫色分割线
            # print("\033[35m" + "=" * 50 + "\033[0m")
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
        # except Exception as e:
        #     print(f"Error processing question: {question}. Error: {e}")
        #     return None  

    def get_final_answer(self, result):
        """
        Extracts the final answer from the RARR result.
        """
        # Assuming 'selected_evidences' contains the final answer after editing
        selected_evidences = result.get("selected_evidences", [])
        if selected_evidences:
            # Implement logic to derive the final answer from selected evidences
            # This is a placeholder and should be adjusted based on actual implementation
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
    # Generate questions for the claim
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
    # print("\033[35m" + "=" * 50 + "\033[0m")
    # Run search on generated question for the claim
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

    # Flatten the evidences per question into a single list.
    # print(evidences_for_questions)
    used_evidences = [
        e
        for cur_evids in evidences_for_questions
        for e in cur_evids[:max_evidences_per_question]
    ]
    # print(used_evidences)
    # Iterative editing over each evidence
    revision_steps = []
    for evid in used_evidences:
        # Run the agreement gate on the current (claim, context, query, evidence) tuple
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

        # Run the editor gate if the agreement gate is open
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

            # Don't keep the edit if the editor makes a huge change
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

def test_and_save(args):
    sys.path.append('/Self-Correction-Benchmark')  # 请根据实际路径修改
    from utils.process_config import open_config
    sys.path.append('/Self-Correction-Benchmark/method_tool/RARR')  # 请根据实际路径修改
    from model import create_model
    from task import create_task

    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    task_config_files = glob.glob(os.path.join(args.task_config_dir, '*.json'))

    if not task_config_files:
        print(f"No task configuration files found in {args.task_config_dir}.")
        return

    for task_config_path in task_config_files:
        task_name = os.path.splitext(os.path.basename(task_config_path))[0]
        print(f"\nProcessing Task: {task_name}")

        task_config = open_config(config_path=task_config_path)
        task = create_task(task_config)
        data = task.get_data()

        method = RARR(model, task, args.prompting_style, correct_iteration=args.correct_iteration)

        results_path = f'/Self-Correction-Benchmark//results_tool/{args.method}/{task.task_name}/'  # 修改为实际路径
        results_file = f'{results_path}/{model.name}_results.json'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        print(f"Making a new file {results_file} to save the result.")

        final_results = []
        correct_number = 0
        total_number = len(data['question'])
        empty_answer_count = 0

        for i in tqdm(range(total_number), desc=f"Processing {task.task_name} questions"):
            question = data['question'][i]
        # i = 1

            answer = data['final_answer'][i]
            # answer = 'B'
            record = method(question, answer)
            final_results.append(record)
            if record and record.get('correct', False):
                correct_number += 1
            if record and record.get('final_answer') is None:
                empty_answer_count += 1
            

            ACC = correct_number / (i + 1 - empty_answer_count) if (i + 1 -     empty_answer_count) > 0 else 0
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
    parser = argparse.ArgumentParser(description="RARR Testing and Saving Script for Multiple Tasks")
    parser.add_argument('--model_config', type=str, default='/Self-Correction-Benchmark/config/model_config/api_LLaMA3.1-8B.json',
                        help='Path to the model configuration file.')
    parser.add_argument('--task_config_dir', type=str, default='/Self-Correction-Benchmark/config/task_gpt3.5',
                        help='Path to the directory containing task configuration files.')
    parser.add_argument('--method', type=str, default='rarr',
                        help='Method name to use.')
    parser.add_argument('--prompting_style', type=str, default='zero-shot-cot',
                        choices=['zero-shot-cot', 'few-shot-cot', 'zero-shot'],
                        help='Prompting style to use.')
    parser.add_argument('--correct_iteration', type=int, default=1, 
                        help='Number of correction iterations.')
    args = parser.parse_args()

    test_and_save(args)

if __name__ == "__main__":
    main()
