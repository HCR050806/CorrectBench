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

    def gen_final_verified_response(self, question, baseline_response, verifications, output):
        final_prompt = (
            "The following is a baseline response and its verification results. "
            "Additionally, consider the following output from a previous analysis.\n"
            f"Initial Question: {question}\n"
            f"Baseline Response: {baseline_response}\n"
            "Verification Results:"
        )
        for result in verifications:
            final_prompt += f"\n{result}"
        final_prompt += (
            f"\nPrevious Output: {output}\n"
            "Please generate a final response by correcting any errors in the baseline response based on the verification results and the previous output.\n"
            "Your final answer should be in the form \\boxed{answer}, at the end of your response."
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

    def __call__(self, question, answer, output):
        is_right = False
        baseline_response = self.gen_baseline_response(question)
        verification_questions = self.plan_verifications(baseline_response)
        verifications = self.execute_verifications(verification_questions)
        final_verified_response = self.gen_final_verified_response(
            question, baseline_response, verifications, output
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


def extract_outputs(json_file_path):
    """
    从给定的 JSON 文件中提取所有 "output" 值并返回一个列表。

    参数:
    - json_file_path (str): JSON 文件的路径。

    返回:
    - List[str]: 包含所有 "output" 值的列表。
    """
    outputs = []
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        # 检查 'results' 键是否存在且是一个列表
        if 'results' in data and isinstance(data['results'], list):
            for idx, result in enumerate(data['results']):
                output_result = result.get('result')
                output = output_result.get('text')
                if output:
                    outputs.append(output)
                else:
                    print(f"警告: 'output' 在结果索引 {idx} 中不存在。")
        else:
            print("错误: JSON 中缺少 'results' 键或其格式不正确。")

    print("提取到的数量为：", len(output))
    return outputs


def test_and_save(args, outputs):
    import sys
    sys.path.append('/Self-Correction-Benchmark')
    from utils.process_config import open_config
    from model import create_model
    from task import create_task

    # 加载模型配置
    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    # 处理单个任务配置文件
    task_config_path = args.task_config_file
    print(f"Processing task config: {task_config_path}")

    task_config = open_config(config_path=task_config_path)
    task = create_task(task_config)
    data = task.get_data()

    # 检查问题数量与输出数量是否匹配
    num_questions = len(data['question'])
    if num_questions != len(outputs):
        print(f"错误: 任务的问题数量 ({num_questions}) 与输出的数量 ({len(outputs)}) 不匹配。")
        return

    cove = COVE(model, task)

    results_path = f'/Self-Correction-Benchmark/results/{args.method}/{task.task_name}/'
    results_file = f'{results_path}/{model.name}_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    print(f"Saving results to {results_file}")

    final_results = []
    correct_number = 0
    total_number = 0
    empty_answer_count = 0

    for idx, (q, a, output) in enumerate(tqdm(
        zip(data['question'], data['final_answer'], outputs),
        total=num_questions,
        desc=f"Processing {task.task_name}"
    )):
        total_number += 1
        try:
            record = cove(q, a, output)
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

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

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

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Method: {args.method}\nTask: {task.task_name}\nModel: {model.name}\nFinal Accuracy: {ACC:.2f}")
    print(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run COVE on a single task and save results.")
    parser.add_argument(
        '--model_config',
        type=str,
        default='/Self-Correction-Benchmark/config/model_config/api_gpt3.5_config.json',
        help='Path to the model configuration file.'
    )
    parser.add_argument(
        '--task_config_file',
        type=str,
        default='/Self-Correction-Benchmark/config/task_config_all/MATH.json',
        help='Path to the task configuration file.'
    )
    parser.add_argument(
        '--outputs_json',
        type=str,
        default='/Self-Correction-Benchmark/results_tool/prompt_to_rarr/MATH/gpt-3.5-turbo_results.json'
    )
    parser.add_argument('--method', type=str, default='prompt_to_rarr_cove', help='Evaluation method to use.')
    args = parser.parse_args()

    # 提取 outputs
    outputs = extract_outputs(args.outputs_json)
    print(f"提取到的 'output' 数量: {len(outputs)}")

    test_and_save(args, outputs)


if __name__ == "__main__":
    main()
