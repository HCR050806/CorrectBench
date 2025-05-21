import re
import os
import json
from tqdm import tqdm
import argparse
import sys
import torch

# 如果需要自定义路径，可根据实际情况修改
sys.path.append('/Self-Correction-Benchmark')


class RCI:
    def __init__(self, model, task=None, prompting_style='zero-shot-cot', correct_iteration=1):
        self.model = model
        self.task = task
        self.prompting_style = prompting_style
        self.correct_iteration = correct_iteration
        self.initial_prompt = self.get_initial_prompt()
        self.critique_prompt = 'Review your previous answer and find problems with your answer.\n\n'
        self.improve_prompt = (
            'Based on the problems you found, improve your answer. '
            'In the form \\boxed{answer}.\n\n'
        )

    def get_initial_prompt(self):
        if self.prompting_style == 'zero-shot-cot':
            return (
                "Let's think step by step. "
                "In the form \\boxed{answer}, at the end of your response.\n\nA:"
            )
        elif self.prompting_style == 'few-shot-cot':
            return "A:\n"  # TODO: 需要在此替换成你真正的few-shot提示
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

    def get_answer(self, output: str):
        """
        从模型输出中提取 \boxed{...} 之间的文本，并尝试转为数字或保留字符串。
        """
        answer = re.findall(r'\\boxed{(.+?)}', output)
        if answer:
            try:
                return int(answer[0])
            except ValueError:
                try:
                    return float(answer[0])
                except ValueError:
                    return answer[0]  
        else:
            return None  

    def correct(self, initial_input: str, output: str):
        """
        使用 critique_prompt 生成改进提示，再使用 improve_prompt 改写答案。
        """
        critique_input = initial_input + output + '\n\n' + self.critique_prompt
        critique_output = self.model.query(critique_input)
        improve_input = critique_input + '\n\n' + critique_output + '\n\n' + self.improve_prompt
        improve_output = self.model.query(improve_input)
        return critique_output, improve_output

    def __call__(self, question: str, answer: str, final_verified_response: str = None):
        """
        每轮QA流程：
          1. 构造 initial_input，将 question 与 final_verified_response(如果存在) 结合
          2. 生成并记录 round_0 输出
          3. 进行多个迭代纠正 (correct_iteration 次)
          4. 返回最终记录
        """
        try:
            # 如果有 final_verified_response，就把它追加到问题后面
            if final_verified_response:
                # 你也可以选择在中间加个分隔符，如 \n 或空格等
                initial_input = f"Q: {question} {final_verified_response}\n\n{self.initial_prompt}"
            else:
                initial_input = f"Q: {question}\n\n{self.initial_prompt}"

            output = self.model.query(initial_input)
            record = {}
            record['round_0'] = output

            for iter_num in range(self.correct_iteration):
                critique, output = self.correct(initial_input, output)
                record[f'round_{iter_num + 1}'] = {'critique': critique, 'output': output}

                final_answer = self.get_answer(output)
                record['final_answer'] = final_answer
                record['correct answer'] = answer
                record['question'] = question
                print("-----------------------------------------")
                print(f"Predicted answer: {final_answer}")
                print(f"Correct answer: {answer}")
                print("-----------------------------------------")

                if str(final_answer) == answer:
                    record['correct'] = True
                    break  
                elif str(final_answer) == answer[1:]:  # 可选逻辑：去掉前缀等特殊处理
                    record['correct'] = True
                    break 
                else:
                    record['correct'] = False

            if not record.get('correct', False):
                record['error'] = 'Final answer and answer do not match'

            return record
        except Exception as e:
            print(f"Error processing question: {question}. Error: {e}")
            return None  

def extract_final_verified_responses(json_file_path: str):
    """
    读取指定的 JSON 文件，提取每个结果中的 'Final Verified Response'，
    并将其存储在一个列表中返回。
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        final_verified_responses = []
        # 假设顶层有 "results" 键，里面是列表，每个元素可能包含 "Final Verified Response"
        if 'results' in data and isinstance(data['results'], list):
            for item in data['results']:
                response = item.get('Final Verified Response')
                if response:
                    final_verified_responses.append(response)
                else:
                    final_verified_responses.append(None)
        else:
            print("JSON 数据中没有 'results' 键或 'results' 不是一个列表。")
        
        return final_verified_responses

    except FileNotFoundError:
        print(f"文件未找到: {json_file_path}")
        return []
    except json.JSONDecodeError:
        print(f"文件不是有效的 JSON 格式: {json_file_path}")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

def test_and_save(args):
    """
    主要流程：
      1. 加载并初始化模型
      2. 打开并读取指定的 task_config（只处理单一任务）
      3. 如果提供了 verified_responses（default 已设好），提取其中的 Final Verified Response
      4. 用 RCI 类进行推理和修正
      5. 保存结果到 /Self-Correction-Benchmark/results/... 目录下
    """
    # 以下 import 根据实际项目结构调整
    from utils.process_config import open_config
    from model import create_model
    from task import create_task

    # 1) 加载模型配置并创建模型
    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    # 2) 读取任务配置
    task_config_path = args.task_config
    if not os.path.isfile(task_config_path):
        print(f"Task configuration file not found: {task_config_path}")
        return

    task_name = os.path.splitext(os.path.basename(task_config_path))[0]
    print(f"\nProcessing Task: {task_name}")

    task_config = open_config(config_path=task_config_path)
    task = create_task(task_config)
    data = task.get_data()  # 应返回 {"question": [...], "final_answer": [...]}

    # 3) 提取 verified_responses（Final Verified Response）
    verified_responses_list = []
    if args.verified_responses and os.path.isfile(args.verified_responses):
        verified_responses_list = extract_final_verified_responses(args.verified_responses)
        # 若数量与 question 不符，这里可做截断或其他处理
        if len(verified_responses_list) != len(data['question']):
            print("警告: 题目数量与 verified_responses 数量不匹配，进行截断处理。")
            min_len = min(len(verified_responses_list), len(data['question']))
            verified_responses_list = verified_responses_list[:min_len]
            data['question'] = data['question'][:min_len]
            data['final_answer'] = data['final_answer'][:min_len]

    # 4) 初始化 RCI 方法，并进行推理
    method = RCI(
        model=model,
        task=task,
        prompting_style=args.prompting_style,
        correct_iteration=args.correct_iteration
    )

    results_path = f'/Self-Correction-Benchmark/results/{args.method}/{task.task_name}/'
    results_file = f'{results_path}/{model.name}_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    print(f"Making a new file {results_file} to save the result.")

    final_results = []
    correct_number = 0
    total_number = len(data['question'])
    empty_answer_count = 0

    for i in tqdm(range(total_number), desc=f"Processing {task.task_name} questions"):
        question = data['question'][i]
        answer = data['final_answer'][i]
        # 如果有 verified_responses，就取第 i 个
        response_for_this_question = verified_responses_list[i] if i < len(verified_responses_list) else None

        record = method(question, answer, final_verified_response=response_for_this_question)
        final_results.append(record)

        if record is not None and record.get('correct', False):
            correct_number += 1
        if record is not None and record.get('final_answer') is None:
            empty_answer_count += 1

        # 计算当前 ACC
        attempts_so_far = i + 1 - empty_answer_count
        ACC = correct_number / attempts_so_far if attempts_so_far > 0 else 0

        results_dict = {
            "ACC": ACC,
            "empty_answers": empty_answer_count,
            "results": final_results
        }
        # 每处理一个问题就保存一次中间结果
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)

    print(f"Method: {args.method}")
    print(f"Task: {task.task_name}")
    print(f"Model: {model.name}")
    print(f"Final Accuracy: {ACC:.2f}")
    print(f"Number of questions with empty answers: {empty_answer_count}")
    print(f"Results saved to {results_file}")


def main():
    """
    使用示例（如果不传参会使用默认）：
    python your_script.py

    或者手动指定参数：
    python your_script.py \
        --task_config /your/task_config.json \
        --verified_responses /your/verified_responses.json
    """
    parser = argparse.ArgumentParser(
        description="RCI Testing and Saving Script for MATH Task with Verified Responses"
    )
    parser.add_argument(
        '--model_config',
        type=str,
        default='/Self-Correction-Benchmark/config/model_config/LLaMA3.1-8B.json',
        help='Path to the model configuration file.'
    )
    parser.add_argument(
        '--task_config',
        type=str,
        default='/Self-Correction-Benchmark/config/task_config/MATH.json',
        help='Path to the MATH task configuration file.'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='cove_to_rci',
        help='Method name to use.'
    )
    parser.add_argument(
        '--prompting_style',
        type=str,
        default='zero-shot-cot',
        choices=['zero-shot-cot', 'few-shot-cot', 'zero-shot'],
        help='Prompting style to use.'
    )
    parser.add_argument(
        '--correct_iteration',
        type=int,
        default=1,
        help='Number of correction iterations.'
    )
    parser.add_argument(
        '--verified_responses',
        type=str,
        default='/Self-Correction-Benchmark/method_mixture/Llama-3.1-8B-Instruct_results.json',
        help='Path to the JSON file containing Final Verified Responses.'
    )

    args = parser.parse_args()
    test_and_save(args)


if __name__ == "__main__":
    main()
