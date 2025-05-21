import pprint
import json
import re
import numpy as np
import random
from math import isclose
from typing import Union, Any

def round_with_error(x):
    return round(x * 1e5) / 1e5


def floatify_ans(ans):
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
            ans = round_with_error(ans)
        except Exception:
            ans = str(ans)
    return ans

def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision

def normalize_answer(answer: str):
    answer = str(answer)
    # number
    answer = answer.replace(",", "")
    digits = re.findall(r"-?\d+\.?\d*", answer)
    answer = digits[-1] if len(digits) > 0 else None
    return floatify_ans(answer)

def finqa_equal(prediction: Union[bool, float, str],
                reference: Union[float, str],
                include_percentage: bool = True,
                is_close: float = False) -> bool:
    #print("prediction:",prediction)
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
                precision = min(get_precision(prediction), get_precision(item))
                if round(prediction, precision) == round(item, precision):
                    return True
            except Exception:
                continue
        return False

def get_metrics(file_path, oracle=False):
    
    scores = []
    with open(file_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        for idx, sample in enumerate(data):
            #sample = json.loads(line)
            preds = []

            # for inference result evaluation
            if not isinstance(sample['pred'], list):
                sample['pred'] = [normalize_answer(sample['pred'])]

            # if None use previous answer
            for p in sample['pred']:
                preds.append(preds[-1] if (p is None and preds) else p)
                #print("sanple",p)
            is_correct = [finqa_equal(p, sample['gt']) for p in preds]

            if oracle: # critic(oracle): only revise incorrect answer
                stop_idx = next((i for i, c in enumerate(is_correct) if c), None)
            else: # critic: stop if no correction twice (double check)
                stop_idx = next((i for i in range(2, len(preds)) if preds[i] == preds[i-1] == preds[i-2]), None)

            if stop_idx is not None:
                is_correct[stop_idx+1:] = [is_correct[stop_idx]] * (len(is_correct) - stop_idx - 1)

            scores.append(is_correct)
    
    print("num of samples:", len(scores))

    # output mean of each column of scores
    col_means= np.array(scores).mean(axis=0)
    print(list(np.round(col_means * 100, decimals=1)))
    print()


if __name__ == "__main__":
    ## text-davinci-003
    with open('results_filename2.txt', 'r', encoding='utf-8') as f:
        file_path = f.read()
    #file_path = "results\program_critic\gsm8k\Meta-Llama-3.1-70B-Instruct-Turbo_results.json"
    print(file_path)
    print("CRITIC:")
    get_metrics(file_path, oracle=False)
    print("CRITIC (oracle):")
    get_metrics(file_path, oracle=True)
