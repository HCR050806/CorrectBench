import re
import json
import os
from tqdm import tqdm
    
class SELF_REFINE:
    def __init__(self, model, task, init_prompt_examples_file, fb_prompt_examples_file):
        self.model = model
        self.task = task
        self.init_prompt_examples_file = init_prompt_examples_file
        self.fb_prompt_examples_file = fb_prompt_examples_file

    def setup_prompt_from_examples_file(self, init_prompt_examples) -> str:
        with open(init_prompt_examples, "r") as f:
            prompt = f.read()
        return prompt
    
    def make_query1(self, question, init_prompt_examples_file):
        question_prefix = "# Q: "
        answer_prefix = "# solution using Python:\n"
        intra_example_sep = "\n"
        question = question.strip()
        prompt = self.setup_prompt_from_examples_file(init_prompt_examples_file)
        query1 = f"{prompt}{question_prefix}{question}{intra_example_sep}{answer_prefix}"
        return query1

    def make_query2(self, solution, init_prompt_examples_file):
        intra_example_sep = "\n\n"
        instruction = (
            "# There is an error in the code above because of lack of understanding of the question. "
            "What is the error? To find the error, go through semantically complete blocks of the code,"
            "and check if everything looks good."
        )
        prompt = self.setup_prompt_from_examples_file(init_prompt_examples_file)
        query2 = f"{prompt}{solution}{intra_example_sep}{instruction}"
        return query2

    def task_init(self, solution, init_prompt_examples_file):
        generation_query = self.make_query1(solution, init_prompt_examples_file)
        output = self.model.query(generation_query)
        return output

    def task_feedback(self, solution, fb_prompt_examples_file):
        generation_query = self.make_query2(solution, fb_prompt_examples_file)
        output = self.model.query(generation_query)

        if "### END" in output:
            output = output.split("### END")[0]

        if "def solution():" in output:
            improved_soln = output.split("def solution():")[1]
        else:
            improved_soln = None

        feedback = output.split("def solution():")[0]
        if improved_soln is not None:
            improved_soln = "def solution():" + improved_soln.rstrip()
        else:
            improved_soln = "def solution():\n# No solution found"

        return {"solution": improved_soln, "feedback": feedback}

    def __call__(self, question):

        n_attempts = 0
        log = []
        solution = ""
        while n_attempts < 10:
            if n_attempts == 0:
                solution = self.task_init(question, self.init_prompt_examples_file)
            fb_and_maybe_soln = self.task_feedback(solution, self.fb_prompt_examples_file)

            log.append({
                "attempt": n_attempts,
                "solution_curr": solution,
                "solution_fixed": fb_and_maybe_soln["solution"],
                "feedback": fb_and_maybe_soln["feedback"],
            })

            if "it is correct" in fb_and_maybe_soln["feedback"].lower():
                break

            solution = fb_and_maybe_soln["solution"]
            n_attempts += 1

        return log


def test_and_save(args):

    import sys
    sys.path.append("/Self-Correction-Benchmark")
    from utils.process_config import open_config
    from model import create_model
    from task import create_task

    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)
    task_config = open_config(config_path=args.task_config)
    task = create_task(task_config)
    data = task.get_data()

    correction_method = SELF_REFINE(
        model,
        task,
        args.init_prompt_examples_file,
        args.fb_prompt_examples_file
    )

    results_path = f"/Self-Correction-Benchmark/results/{args.method}/{task.task_name}/"
    results_file = f"{results_path}/{model.name}_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    final_results = []
    correct_number = 0
    total_number = 0

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            try:
                saved_data = json.load(f)
                final_results = saved_data.get("results", [])
                correct_number = sum(1 for r in final_results if r["correct"])
                total_number = len(final_results)
            except json.JSONDecodeError:
                pass

    for question, final_answer in tqdm(zip(data["question"], data["final_answer"]), total=len(data["question"])):
        total_number += 1

        log = correction_method(question)

        is_correct = False
        if len(log) > 0:
            if "it is correct" in log[-1]["feedback"].lower():
                is_correct = True

        if is_correct:
            correct_number += 1

        record = {
            "question": question,
            "final_answer": final_answer,
            "correct": is_correct,
            "log": log,
        }
        final_results.append(record)

        # 实时计算并保存
        ACC = correct_number / total_number if total_number > 0 else 0
        output = {
            "ACC": ACC,
            "results": final_results
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        type=str,
        default="/Self-Correction-Benchmark/config/model_config/LLaMA3.1-70B.json"
    )
    parser.add_argument(
        "--task_config_dir",
        type=str,
        default="/Self-Correction-Benchmark/config/test_selfrefine",
        help="Path to your task config directory containing multiple JSON files"
    )
    parser.add_argument("--method", type=str, default="selfrefine")
    parser.add_argument(
        "--init_prompt_examples_file",
        type=str,
        default="/Self-Correction-Benchmark/method/init.txt"
    )
    parser.add_argument(
        "--fb_prompt_examples_file",
        type=str,
        default="/Self-Correction-Benchmark/method/feedback.txt"
    )

    args = parser.parse_args()

    task_config_pattern = os.path.join(args.task_config_dir, "*.json")
    task_config_files = glob.glob(task_config_pattern)

    if not task_config_files:
        print(f"No task config files found in directory: {args.task_config_dir}")
        exit(1)

    for config_path in task_config_files:
        print(f"Processing task config: {config_path}")
        args.task_config = config_path
        test_and_save(args)

    print("All tasks processed.")
