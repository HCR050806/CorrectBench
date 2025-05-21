import re
import os
import json
import func_timeout
from typing import Union, Any
from math import isclose

class Program_Inference:
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
            prompt_path = "/Self-Correction-Benchmark/dataset/GSM8k/few_shot_self_refine_2.txt"
            with open(prompt_path, 'r', encoding='utf-8') as fp:
                demo_prompt = fp.read().strip() + "\n\n"
            #full_prompt = demo_prompt + f'Question: {question}' + '\n'
            self.initial_prompt = demo_prompt + '# Python code, return answer.' + '\n'
        elif self.prompting_style == 'zero-shot':
            self.initial_prompt = "Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.\nA:\n"
        else:
            print("WARNING: The prompting style is not given. Use zero-shot-cot as default.")
            self.initial_prompt = "Let's think step by step. Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.\nA:\n"

    def round_with_error(self,x):
        return round(x * 1e5) / 1e5
    
    def get_answer(self, ans):
        if self.task == None:  # default task is gsm8k
            answer = re.findall(r'\\boxed{(.+?)}', ans)
            return int(answer[0])
        else:
            '''TODO: add the get_answer function for other tasks'''
            if self.task.task_name == 'gsm8k': 
                """gsm8k"""
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
            
    def get_precision(self,gt_ans: float) -> int:
        precision = 5
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision        
            
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

    def correct(self, initial_input, output):
        critique_input = initial_input + output + '\n\n' + self.cririque_promtp
        critique_output = self.model.query(critique_input)
        improve_input = critique_input + '\n\n' + critique_output + '\n\n' + self.improve_prompt
        improve_output = self.model.query(improve_input)
        return critique_output, improve_output
    
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

        
    def __call__(self, question, answer):
        correct = 0
        wrong = 0
        initial_input = 'Q: ' + question + '\n\n' + self.initial_prompt
        output = self.model.query(initial_input)
        output = output.split('```python',1)[1].split('```')[0].replace('Question:','# Question:')
        print("output:",output)

        print("\ncontent!!\n")
        print(output)
        ans, report = self.safe_execute(output)
        #提取各种类型的ans
        prediction = self.get_answer(ans)
        gt_cot, gt_ans = answer.split("####") # GSM8k
        gt_cot, gt_ans = gt_cot.strip(), self.get_answer(gt_ans.strip())
        is_correct = self.finqa_equal(prediction, gt_ans)
        if is_correct:
            correct += 1
        else:
            wrong += 1

        sample = {'question': question, 'gt_cot': gt_cot, 'gt': gt_ans,
               'pred': prediction}
        sample.update({'report': report, 'code': output})
        return sample
        



'''A test function for the class of Program_Inference'''
def test():
    import sys
    #sys.path.append('/mnt/yuanzenghui/Self-Correction-Benchmark')
    #sys.path.append('/Self-Correction-Benchmark')
    #parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parent_dir = "/Self-Correction-Benchmark"
    print(parent_dir)
    sys.path.append(parent_dir)
    from utils.process_config import open_config
    from model import create_model
    from task import create_task
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    #/mnt/yuanzenghui/Self-Correction-Benchmark/config/model_config/api_llama_config.json
    # parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # sys.path.append(parent_dir)

    parser.add_argument('--start_task', type=int, default=0)
    parser.add_argument('--end_task', type=int, default=100)
    parser.add_argument('--model_config', type=str, default='/Self-Correction-Benchmark/config/model_config/api_deepseek_config.json')
    parser.add_argument('--task_config', type=str, default='/Self-Correction-Benchmark/config/task_config/gsm.json')
    parser.add_argument('--method', type=str, default='program_inference')
    parser.add_argument('--prompting_style', type=str, default='few-shot-cot')
    args = parser.parse_args()

    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    task_config = open_config(config_path=args.task_config)
    task = create_task(task_config)
    data = task.get_data()

    '''Create a directory to store the results'''
    results_path = f'results/{args.method}/{task.task_name}/'
    results_file = f'{results_path}/{model.name}_results_{args.start_task}_{args.end_task}.json'
    with open('results_filename2.txt', 'w', encoding='utf-8') as f:
        f.write(results_file)
    dic = os.path.dirname(results_file)
    if not os.path.exists(dic):
        os.makedirs(dic)
    with open(results_file, 'w') as f:
        json.dump({}, f)
    print(f"Make a new file {results_file} to save the inference result.")

    #inference
    inference_result = []
    idx = args.start_task
    print(data['question'][args.start_task:args.end_task])
    #return 
    for q, a in tqdm(zip(data['question'][args.start_task:args.end_task], data['answer'][args.start_task:args.end_task])):
        print("idx: ",idx)
        # if idx < args.start_task or (args.end_task != -1 and idx >= args.end_task):
        #     continue
        # if idx >= args.end_task:
        #     break
        #将idx编号加进example里
        #example = {**{'idx': idx}, **example}
        print("question: ",q)
        print("answer: ",a)
        inference_method = Program_Inference(model, task, args.prompting_style)
        record = inference_method(q, a)
        #print("sample: ",record)
        inference_result.append(record)
        idx += 1

    with open(results_file, 'w') as f:
        json.dump(inference_result, f, indent=4)
    # for q, a in tqdm(zip(data['question'], data['final_answer'])):
    #     correction_method = Program_Critic(model, task, args.prompting_style)
    #     record = correction_method(q, a)
    #     print(record)
    #     break
    
if "__main__" == __name__:
    test()
