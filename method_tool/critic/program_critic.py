import re
import os
import json
import func_timeout
from typing import TypeVar, Iterable, List, Union, Any
from pathlib import Path
from math import isclose

class Program_Critic:
    def __init__(self, model, task=None, prompting_style='zero-shot-cot', correct_iteration=1):
        self.model = model
        self.task = task
        self.prompting_style = prompting_style
        self.initial_prompt()
        self.cririque_promtp = 'Review your previous answer and find problems with your answer.\n\n'
        self.improve_prompt = 'Based on the problems you found, improve your answer. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{answer}.\n\n'
        self.correct_iteration = correct_iteration

    def initial_prompt(self):
        if self.prompting_style == 'zero-shot-cot':
            self.initial_prompt = "Let's think step by step. Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.\n\nA:"
        elif self.prompting_style == 'few-shot-cot':
            #self.initial_prompt = "A:\n"   #TODO: add the few-shot prompt file
            prompt_path = "/Self-Correction-Benchmark/dataset/GSM8k/few_shot_self_refine_3.txt"
            with open(prompt_path, 'r', encoding='utf-8') as fp:
                prompt = fp.read().strip() + "\n\n"
            self.initial_prompt = prompt
        elif self.prompting_style == 'zero-shot':
            self.initial_prompt = "Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.\nA:\n"
        else:
            print("WARNING: The prompting style is not given. Use zero-shot-cot as default.")
            self.initial_prompt = "Let's think step by step. Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.\nA:\n"

    def get_answer(self, ans):
        if self.task == None:  # default task is gsm8k
            answer = re.findall(r'\\boxed{(.+?)}', ans)
            return int(answer[0])
        else:
            '''TODO: add the get_answer function for other tasks'''
            if self.task.task_name == 'gsm8k': 
                if ans is None:
                    return None
                elif type(ans) == dict:
                    ans = list(ans.values())[0]
                elif type(ans) == bool:
                    ans = ans
                elif type(ans) in [list, tuple]:
                    if not ans:
                        return None
                    else:
                        try:
                            ans = float(ans[0])
                        except Exception:
                            ans = str(ans[0])
                else:
                    try:
                        ans = float(ans)
                        ans = self.round_with_error(ans)
                    except Exception:
                        ans = str(ans)
                return ans
            else:
                raise ValueError("The task name is not supported.")

    def correct(self, initial_input, output):
        critique_input = initial_input + output + '\n\n' + self.cririque_promtp
        critique_output = self.model.query(critique_input)
        improve_input = critique_input + '\n\n' + critique_output + '\n\n' + self.improve_prompt
        improve_output = self.model.query(improve_input)
        return critique_output, improve_output

    def remove_comment(self,code):
        code = code.split("\n")
        code = [line for line in code if not line.startswith("#")]
        code = [line for line in code if line.strip() != ""]
        return "\n".join(code)    

    def safe_execute(self,code_string: str, keys=None):
        def execute(x):
            try:
                exec(x)
                locals_ = locals()
                if keys is None:
                    an = locals_.get('answer', None)
                else:
                    an = [locals_.get(k, None) for k in keys]
                return an, "Done"
            except BaseException as e: # jump wrong case
                return None, repr(e)

        try:
            an, report = func_timeout.func_timeout(3, execute, args=(code_string,))
        except func_timeout.FunctionTimedOut:
            an = None
            report = "TimeoutError: execution timeout"

        return an, report
    
    def finqa_equal(self,prediction: Union[bool, float, str],
                    reference: Union[float, str],
                    include_percentage: bool = True,
                    is_close: float = False) -> bool:
        if prediction is None:
            return False
        elif type(prediction) == bool:
            # bool questions
            if prediction:
                return reference == 'yes'
            else:
                return reference == 'no'
        elif type(reference) == str or type(prediction) == str:
            # string questions
            return prediction == reference
        else:
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=0.001):
                            return True
                    precision = min(self.get_precision(prediction), self.get_precision(item))
                    if round(prediction, precision) == round(item, precision):
                        return True
                except Exception:
                    continue
            return False
    def __call__(self, sample):
        # initial_input = 'Q: ' + question + '\n\n' + self.initial_prompt
        # output = self.model.query(initial_input)
        # record = {}
        # record['round_0'] = output
        # for iter in range(self.correct_iteration):
        #     critique, output = self.correct(initial_input, output)
        #     record['round_'+str(iter+1)] = {'critique': critique, 'output': output}
        # final_answer = self.get_answer(output)
        # record['final_answer'] = final_answer
        # if final_answer == int(answer):
        #     record['correct'] = True
        # else:
        #     record['correct'] = False
        # return record
        prompt = 'Q: ' + sample['question'] + '\n\n' + self.initial_prompt
        max_iter = 2
        for itr in range(1, max_iter + 1):
            if itr == 1:
                print("Is initial program correct:", sample['gt'] == sample['pred'])
                sample['pred'] = [sample['pred']]
                sample['report'] = [sample['report']]
                sample['code'] = [sample['code']]
            print("\n" + "-" * 20, "iteration", itr, "-" * 20)
            
            # criticize latest answer that is not "None"
            base_idx = itr - 1
            while base_idx > 0 and sample['pred'][base_idx] is None:
                base_idx -= 1
            print("Correct based on iter:", base_idx)

            previous_code = self.remove_comment(sample['code'][base_idx])

            # construct prompt
            context = f"Question: {sample['question']}\n"
            context += f"```python\n{previous_code}\n```\n"

            context += f"Execution: {sample['report'][base_idx]}\n"
            context += f"Output: answer = {self.get_answer(sample['pred'][base_idx])}\n"
            context += "\nWhat's the problem with the above code? If you think it is correct, just outout the code.\n\n"
            prompt_critic = prompt + context
            print(context, end="")

            output = self.model.query(prompt_critic)
            if '```python' in output and '```' in output:
                output = output.strip().split('```python',1)[1].split('```')[0]
            else :
                output = output.strip()    
            print("output:",output)

            # if context not end with a "\n", add "\n"
            if context and context[-1] != "\n":
                context += "\n"

            # generate new code
            context += "Here's a better solution:\n```python\n"
            prompt_critic += context
            print(context, end="")

            output = self.model.query(prompt_critic)
            print("output: ",output)
            # excute new code
            print("\ncontent!!\n")
            if '```python' in output and '```' in output:
                code = output.strip().split('```python',1)[1].split('```')[0]
            else :
                code = output.strip()    
            #code = output.strip().split('```python',1)[1].split('```')[0]

            print("!!!!!!!: \n",code)
            pred, report = self.safe_execute(code)
            pred = self.get_answer(pred)
            corrected = True
            print("{}\n```".format(code))
            print("Execution:", report)
            print("Output: answer =", pred)

            if code.strip() == sample['code'][base_idx].strip(): # no correction
                corrected = False
                code = sample['code'][base_idx]
                report = sample['report'][base_idx]
                pred = sample['pred'][base_idx]

            # append new result
            sample['code'].append(code)
            sample['report'].append(report)
            sample['pred'].append(pred)
            is_correct = self.finqa_equal(str(pred), str(sample['gt']))

            print("Gold Answer:", sample['gt'])
            print("Corrected:", "Yes" if corrected else "No")
            print("Is correct:", is_correct)

        # writer.write(json.dumps(sample) + '\n')
        # writer.flush()  
        return sample      




def test():
    import sys
    parent_dir = "/Self-Correction-Benchmark"
    sys.path.append(parent_dir)
    from utils.process_config import open_config
    from model import create_model
    from task import create_task
    from tqdm import tqdm
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_task', type=int, default=0)
    parser.add_argument('--end_task', type=int, default=3)
    parser.add_argument('--model_config', type=str, default='/Self-Correction-Benchmark/config/model_config/api_deepseek_config.json')
    parser.add_argument('--task_config', type=str, default='/Self-Correction-Benchmark/config/task_config/gsm.json')
    parser.add_argument('--method', type=str, default='critic')
    parser.add_argument('--prompting_style', type=str, default='few-shot-cot')
    args = parser.parse_args()

    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    task_config = open_config(config_path=args.task_config)
    task = create_task(task_config)

    # init_file 是一个文本文件，内含了要加载的 JSON 文件名(路径)
    with open('results_filename2.txt', 'r', encoding='utf-8') as f:
        init_file = f.read().strip()

    results_path = f'/Self-Correction-Benchmark/results_tool/{args.method}/{task.task_name}'
    results_file = f'{results_path}/{model.name}_results.json'
    dic = os.path.dirname(results_file)
    if not os.path.exists(dic):
        os.makedirs(dic)

    print(f"Make a new file {results_file} to save the critic result.")

    # 统计正确数与总数以计算 ACC
    correct_count = 0
    total_count = 0

    critic_result = []

    # 读取 init_file 中的 JSON 数据
    with open(init_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for idx, sample in enumerate(data):
        if idx < args.start_task:
            continue
        if idx >= args.end_task:
            break

        print(sample)
        critic_method = Program_Critic(model, task, args.prompting_style)
        record = critic_method(sample)

        # 仅当 pred 中含有至少 1 个元素时才进行对比（保证不越界）
        # 判断如果 gt 等于 pred[0] 或 pred[1] 或 pred[2] 则 correct = True
        # 注意：pred 里可能没有足够的元素，需要先判断长度
        current_preds = record.get('pred', [])
        gt_value = record.get('gt', None)

        if len(current_preds) > 0 and str(gt_value) in str(current_preds[:3]):
            record['correct'] = True
            correct_count += 1
        else:
            record['correct'] = False

        total_count += 1

        print("sample: ", record)
        critic_result.append(record)

    # 计算 ACC
    acc = correct_count / total_count if total_count > 0 else 0.0
    print(f"ACC: {acc:.4f}")

    # 将 ACC 和详细结果一起写入文件
    result_to_save = {
        "ACC": acc,
        "results": critic_result
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(result_to_save, f, indent=4)

    print(f"Results saved to {results_file}")

            
if "__main__" == __name__:
    test()
