import os
import json
import re
from tqdm import tqdm
import argparse
from customized_call import ChatWithOurServer
import torch

# 初始化 ChatWithOurServer 客户端
client = ChatWithOurServer(base_url="http://0.0.0.0:65430/v1", model='Llama-3.1-8B-Instruct')



class COVE:
    def __init__(self, model, task=None):
        self.model = model
        self.task = task
        # 用于统计空答案数量
        self.empty_answer_count = 0

    def gen_baseline_response(self, question):
        baseline_prompt = (
            "Please generate a response to the following question. "
            "Answer directly and concisely.\nQuestion:" + question
        )
        baseline_response = self.model.query(baseline_prompt)
        return baseline_response

    def plan_verifications(self, baseline_response):
        verification_prompt = (
            "The following is a baseline response to a question. "
            "Please generate a set of verification questions to check the accuracy of each fact in the response. "
            "List each question on a new line, prefixed with a number and a dot (e.g., 1., 2.).\n"
            "Baseline Response:" + baseline_response
        )
        verification_questions = self.model.query(verification_prompt)
        # 使用正则表达式提取以数字和点开头的问题
        verification_questions = re.findall(r'\d+\.\s*(.*?)\?', verification_questions)
        return verification_questions

    def execute_verifications(self, verification_questions):
        execute_prompt = (
            "Please answer the following question independently. "
            "Ensure your answer is not influenced by any previous responses.\nQuestions:\n"
        )
        verifications = []
        for question in verification_questions:
            prompt = execute_prompt + question + "?"
            verification_answer = self.model.query(prompt)
            # 将问题和回答组合
            verifications.append(f"{question}? {verification_answer}")
        return verifications

    def gen_final_verified_response(self, question, baseline_response, verifications):
        final_prompt = (
            "The following is a baseline response and its verification results. "
            "Please generate a final response by correcting any errors in the baseline response based on the verification results.\n"
            f"Initial Question: {question}\n"
            f"Baseline Response: {baseline_response}\n"
            "Verification Results:"
        )
        for result in verifications:
            final_prompt += f"\n{result}"
        final_prompt += (
            "\nYour final answer should be in the form \\boxed{answer}, at the end of your response."
        )
        final_verified_response = self.model.query(final_prompt)
        return final_verified_response

    def get_answer(self, final_verified_response):
        """
        如果正则无法匹配到任何 \\boxed{...} 的内容，则认为答案为空，
        增加 empty_answer_count，并返回空字符串。
        """
        answer = re.findall(r'\\boxed{(.+?)}', final_verified_response)
        if not answer:
            self.empty_answer_count += 1
            return ""
        print('********************************')
        print(answer)
        print('********************************')
        return answer[0]

    def __call__(self, question, answer):
        is_right = False
        baseline_response = self.gen_baseline_response(question)
        verification_questions = self.plan_verifications(baseline_response)
        verifications = self.execute_verifications(verification_questions)
        final_verified_response = self.gen_final_verified_response(
            question, baseline_response, verifications
        )
        predicted_answer = self.get_answer(final_verified_response)

        # 判断正确性
        if predicted_answer == answer[1:]:
            is_right = True
        if predicted_answer == answer:
            is_right = True
        if not predicted_answer:
            is_right = None

        print("------------------------------------------------------")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Correct Answer: {answer}")
        print(type(predicted_answer), type(answer))
        print(len(predicted_answer), len(answer))
        print(is_right)
        print("------------------------------------------------------")

        record = {
            "Question": question,
            "Baseline Response": baseline_response,
            "Verifications": verifications,
            "Final Verified Response": final_verified_response,
            "Predicted Answer": predicted_answer,
            "Correct Answer": answer,
            "Correct": is_right
        }
        return record


def test_and_save(args):
    import sys
    sys.path.append('/Self-Correction-Benchmark')
    from utils.process_config import open_config
    from model import create_model
    from task import create_task

    # 加载模型配置
    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    # 遍历任务配置文件夹中的所有任务配置文件
    task_config_dir = args.task_config_dir
    task_config_files = [
        f for f in os.listdir(task_config_dir)
        if os.path.isfile(os.path.join(task_config_dir, f)) and f.endswith('.json')
    ]

    if not task_config_files:
        print(f"No task config files found in directory: {task_config_dir}")
        return

    for task_config_file in task_config_files:
        task_config_path = os.path.join(task_config_dir, task_config_file)
        print(f"Processing task config: {task_config_path}")


        task_config = open_config(config_path=task_config_path)
        task = create_task(task_config)
        data = task.get_data()

        cove = COVE(model, task)

        results_path = f'/Self-Correction-Benchmark/{args.method}/{task.task_name}/'
        results_file = f'{results_path}/{model.name}_results.json'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        print(f"Saving results to {results_file}")

        final_results = []
        correct_number = 0
        total_number = 0
        empty_answer_count = 0

        for q, a in tqdm(
            zip(data['question'], data['final_answer']),
            total=len(data['question']),
            desc=f"Processing {task.task_name}"
        ):
            total_number += 1
            try:
                record = cove(q, a)
                final_results.append(record)

                if record["Correct"]:
                    correct_number += 1
                if record["Predicted Answer"] == "":
                    empty_answer_count += 1


                ACC = correct_number / (total_number - empty_answer_count) if (total_number - empty_answer_count) > 0 else 0.0


                output_data = {
                    "ACC": ACC,
                    "empty_answers": empty_answer_count,
                    "results": final_results
                }

                with open(results_file, 'w') as f:
                    json.dump(output_data, f, indent=4)

            except Exception as e:
                print(f"Error processing question: {q}\nError: {e}")
                record = {
                    "Question": q,
                    "Error": str(e)
                }
                final_results.append(record)

                empty_answer_count += 1
                ACC = correct_number / (total_number - empty_answer_count) if (total_number - empty_answer_count) > 0 else 0.0

                output_data = {
                    "ACC": ACC,
                    "empty_answers": empty_answer_count,
                    "results": final_results
                }

                with open(results_file, 'w') as f:
                    json.dump(output_data, f, indent=4)

        print(f"Method: {args.method}\nTask: {task.task_name}\nModel: {model.name}\nFinal Accuracy: {ACC:.2f}")
        print(f"Results saved to {results_file}")



def main():
    parser = argparse.ArgumentParser(description="Run COVE on multiple tasks and save results.")
    parser.add_argument(
        '--model_config',
        type=str,
        default='/Self-Correction-Benchmark/config/model_config/api_LLaMA3.1-70B.json'
    )
    parser.add_argument(
        '--task_config_dir',
        type=str,
        default='/Self-Correction-Benchmark/config/task_test'
    )
    parser.add_argument('--method', type=str, default='cove')
    args = parser.parse_args()

    test_and_save(args)


if __name__ == "__main__":
    main()
