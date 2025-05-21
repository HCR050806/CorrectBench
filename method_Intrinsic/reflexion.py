# main.py
from typing import List
import tiktoken
import json
import os
from transformers import Qwen2Tokenizer
from reflexion_env import QAEnv, normalize_answer

from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS
from prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER
from customized_call import ChatWithOurServer

client = ChatWithOurServer(base_url="http://0.0.0.0:65430/v1", model="Llama-3.1-8B-Instruct")


class REFLEXION:
    def __init__(
        self,
        model,
        task: None,
        env: QAEnv,
    ) -> None:
        self.model = model
        self.env = env(model)
        self.env.reset()
        self.reset()
        self.agent_prompt = react_reflect_agent_prompt
        self.reflect_prompt = reflect_prompt
        self.enc = tiktoken.encoding_for_model("text-davinci-003")
        self.reflections = []

    def run(self, question, key, reset=True) -> None:
        self.question = question
        self.key = key
        self.env.question = question
        self.env.key = key

        if (self.is_terminated() or self.is_truncated()) and not self.is_correct():
            self.reflect()

        if reset:
            self.env.reset()
            self.reset()

        while not (self.is_truncated() or self.is_terminated()):
            self.step()

    def step(self) -> None:
        self.scratchpad += f"\nThought {self.curr_step}:"
        self.scratchpad += " " + self.prompt_agent()
        # print("-" * 80)
        # print(self.scratchpad)

        self.scratchpad += f"\nAction {self.curr_step}:"
        action = self.prompt_agent()
        self.scratchpad += " " + action
        # print("-" * 80)

        self.scratchpad += f"\nObservation {self.curr_step}: "
        observation, self.reward, self.terminated, self.truncated, self.curr_step = self.env.step(action)
        self.scratchpad += observation
        # print("-" * 80)

    def reflect(self) -> None:
        rflxion = self.prompt_reflection()
        self.scratchpad += f"\nReflexion: {rflxion}"
        self.reflections.append(rflxion)

    def prompt_reflection(self) -> str:
        prmpt = self._build_reflection_prompt()
        # print("\033[91m" + "-" * 80 + "\033[0m")
        # print(prmpt)
        # print("\033[92m" + "-" * 80 + "\033[0m")
        query = self.model.query(prmpt)
        if "\n" in query:
            query = query.split("\n", 1)[0]
        # print(query)
        # print("\033[93m" + "-" * 80 + "\033[0m")
        return self.format_step(query)

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=REFLECTIONS,
            question=self.question,
            scratchpad=self._format_scratchpad()
        )

    def prompt_agent(self) -> str:
        query = self.model.query(self._build_agent_prompt())
        if "\n" in query:
            query = query.split("\n", 1)[0]
        return self.format_step(query)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=WEBTHINK_SIMPLE6,
            reflections=self.format_reflections(self.reflections),
            question=self.question,
            scratchpad=self.scratchpad
        )

    def _format_scratchpad(self) -> str:
        lines = self.scratchpad.split("\n")
        lines_by_tokens = sorted(lines, key=lambda x: len(self.enc.encode(x)))
        
        # 当整个scratchpad的编码长度大于1600时，进入修剪循环
        while len(self.enc.encode("\n".join(lines))) > 1600:
            # 确保 lines_by_tokens 不为空
            if lines_by_tokens:
                # 弹出 lines_by_tokens 中的最后一项
                ind = lines.index(lines_by_tokens.pop(-1))
                line = lines[ind]
                # 将该行修改为简短的表示形式
                lines[ind] = line.split(":")[0] + ": ..."
            else:
                # 如果 lines_by_tokens 为空，则跳出循环，避免死循环
                print("Warning: lines_by_tokens is empty, skipping truncation.")
                break
        
        # 返回处理后的字符串
        return "\n".join(lines)

    def is_terminated(self) -> bool:
        return self.env.is_terminated()

    def is_correct(self) -> bool:
        return self.env.is_correct()

    def is_truncated(self) -> bool:
        return self.env.is_truncated() or (len(self.enc.encode(self._build_agent_prompt())) > 3896)

    def reset(self) -> None:
        self.scratchpad = ""
        self.curr_step = 1

    def format_reflections(self, reflections: List[str]) -> str:
        if not reflections:
            return ""
        else:
            header = REFLECTION_HEADER
            return header + "Reflections:\n- " + "\n- ".join([r.strip() for r in reflections])

    def format_step(self, step: str) -> str:
        return step.strip("\n").strip().replace("\n", "")

    def __call__(self, question, answer, reset=True):
        self.env(question, answer)
        self.run(question, answer, reset=reset)


def test_and_save(model_config_path, task_config_path, method):
    import sys
    sys.path.append("E:/Self-Correction-Benchmark")
    from utils.process_config import open_config
    from model import create_model
    from task import create_task
    from tqdm import tqdm

    model_config = open_config(config_path=model_config_path)
    model = create_model(model_config)
    task_config = open_config(config_path=task_config_path)
    task = create_task(task_config)
    data = task.get_data()

    correction_method = REFLEXION(model, task, QAEnv)

    task_name = task.task_name
    model_name = model.name
    results_path = f"E:/Self-Correction-Benchmark/results_reflexion/{method}/{task_name}/"
    results_file = f"{results_path}/{model_name}_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    empty_answers_count = 0

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            existing_data = json.load(f)
        final_results = existing_data.get("results", [])
        correct_number = sum(1 for r in final_results if r.get("correct", False))

        empty_answers_count = existing_data.get("EmptyAnswersCount", 0)
    else:
        final_results = []
        correct_number = 0

    total_number = len(data["question"])
    current_index = len(final_results)

    for q, a in tqdm(zip(data["question"], data["final_answer"]), total=total_number, initial=current_index):
        # print("yes")
        n = 2
        logs = {"Question": q, "Trials": []}

        correction_method.env.reset()
        correction_method.reset()
        correction_method.reflections = []

        for i in range(n):
            if i == 0:
                correction_method(q, a, reset=True)
            else:
                correction_method(q, a, reset=False)

            logs["Trials"].append(
                {
                    f"Trial {i+1}": {
                        "Scratchpad": correction_method.scratchpad,
                        "Correctness": correction_method.env.is_correct(),
                    }
                }
            )

            if correction_method.env.is_correct():
                break

        norm_key = normalize_answer(correction_method.env.key)      
        norm_ans = normalize_answer(correction_method.env.answer)    

        if norm_ans == "":
            empty_answers_count += 1

        record = {
            "question": q,
            "correct_answer": norm_key,   
            "final_answer": norm_ans,     
            "correct": correction_method.env.is_correct(),
            "logs": logs,
        }
        print("-------------------------------")
        print(f"correct_answer:{norm_key}")
        print(f"correct_answer:{norm_ans}")
        print(correction_method.env.is_correct())
        print("-------------------------------")
        final_results.append(record)

        if record["correct"] and norm_ans != "":
            correct_number += 1

        non_empty_count = len(final_results) - empty_answers_count
        if non_empty_count > 0:
            ACC = correct_number / non_empty_count
        else:
            ACC = 0

        output = {
            "ACC": ACC,
            "EmptyAnswersCount": empty_answers_count,
            "results": final_results
        }

        with open(results_file, "w") as f:
            json.dump(output, f, indent=4)

        # print(f"Intermediate result saved. Current ACC (excluding empty answers): {ACC:.2f}")
        # print(f"Empty answers so far: {empty_answers_count}")

    print(f"Method: {method}\nTask: {task_name}\nModel: {model_name}\nFinal Accuracy (non-empty): {ACC:.2f}")
    print(f"Total empty answers: {empty_answers_count}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        type=str,
        default="E:/Self-Correction-Benchmark/config/model_config/api_Qwen2.5-72B_config.json",
        help="Path to your model config JSON",
    )
    parser.add_argument(
        "--task_config_dir",
        type=str,
        default="E:/Self-Correction-Benchmark/config/task_config_qwq72B",
        help="Path to your task config directory containing multiple JSON files",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="reflexion",
        help="Method name, used to build the results path",
    )

    args = parser.parse_args()
    # test_and_save(args.model_config, args.task_config_dir, args.method)
    task_config_pattern = os.path.join(args.task_config_dir, "*.json")
    task_config_files = glob.glob(task_config_pattern)

    if not task_config_files:
        print(f"No task config files found in directory: {args.task_config_dir}")
        exit(1)

    for task_config_path in task_config_files:
        print(f"Processing task config: {task_config_path}")
        test_and_save(args.model_config, task_config_path, args.method)

    print("All tasks processed.")
