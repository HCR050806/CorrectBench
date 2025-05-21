import json
import re
import string
import ast
import numpy as np
import pprint
from collections import Counter
from typing import TypeVar, Iterable, List, Union, Any



def extract_cot_answer(cot):
    if not cot:
        return ""

    # get answer
    cot = cot.strip(" ")
    cot = cot.split("<|endoftext|>")[0]  # text-davinci-003
    TEMPLATE = "is: "
    if TEMPLATE not in cot:
        return ""

    start_idx = cot.rfind(TEMPLATE) + len(TEMPLATE)
    end_idx = -1 if cot.endswith(".") else len(cot)
    ans_span = cot[start_idx: end_idx].strip()

    return ans_span


def is_null_answer(text):
    if not text:
        return True
    text = text.strip().lower()
    if  text in ["none", "", "no answer", "never", "null", "both", "neither"]:
        return True
    if text.startswith("none"):
        return True
    return False


def get_end_index(tokens, end_tokens=["\n", "<|endoftext|>"], verbose=True):
    stop_token = None
    for end_tk in end_tokens:
        for tk in tokens:
            if end_tk in tk:
                stop_token = tk
                break
    if not stop_token:
        end_idx = len(tokens)
    else:
        end_idx = tokens.index(stop_token)
    return end_idx


def normalize_answer(s):

    def replace_ordinals(s):
        ordinal_map = {
            "first": "1",
            "second": "2",
            "third": "3",
            "fourth": "4",
            "fifth": "5",
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            # more as needed
        }
        for ordinal, number in ordinal_map.items():
            s = s.replace(ordinal, number)
        return s

    def normalize_yes_no(s):
        mp = {
            "true": "yes",
            "false": "no",
        }
        return mp.get(s, s)
    
    def remove_rank(text):
        return re.sub(r"(?<=\w)(st|nd|rd|th)\b", "", text)

    def remove_articles(text):
        if len(text.split(" ")) > 1:
            text = re.sub(r"\b(a|an|the)\b", " ", text)
        # remove 's' in the end of word
        text = " ".join([t.rstrip("s") for t in text.split(" ")])
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`", u"–"]))
        for c in exclude:
            text = text.replace(c, " ")
        return text

    def lower(text):
        return text.lower()
    
    def normalize_number(a):
        if a[-1] == ".":
                a = a[:-1]
        if a[-2:] == ".0":
            a = a[:-2]
        return a

    return normalize_yes_no(remove_rank(replace_ordinals(white_space_fix(remove_articles(remove_punc(lower(s)))))))


def em_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    em = (normalized_prediction == normalized_ground_truth)

    ZERO_METRIC = (0, 0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, f1, precision, recall


def multi_ref_score(prediction, ground_truth):
    """
    ground_truth: list or str
        - multiple references split by "; "
    """
    if isinstance(ground_truth, str):
        ground_truth = ground_truth.split("; ")
    scores = [em_f1_score(prediction, gt)[:2] for gt in ground_truth]
    scores.append(em_f1_score(prediction, " ".join(ground_truth))[:2]) # may predict multi answer
    # scores = sorted(scores, reverse=True)
    return max(scores)





def rejection_sampling_eval(file_path, n=10):
    all_scores = []
    with open(file_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        for idx, sample in enumerate(data):
            if "prediction" not in sample or "temperature_0.5" not in sample['prediction']:
                continue
            sampled_preds = sample['prediction']['temperature_0.5']['text'][:n]

            # extract cot answer
            if "cot" in file_path:
                sampled_preds = [extract_cot_answer(p) for p in sampled_preds]
            
            scores = [multi_ref_score(pred, sample['answer']) for pred in sampled_preds]

            # get max scores: best-of-n
            best_score = max(scores)
            all_scores.append(best_score)

        # average score of all samples
        average_score = np.array(all_scores).mean(axis=0)
        em, f1 = list(np.round(average_score * 100, decimals=1))
        print(f"Best-of-{n}: {em} & {f1}")


def evaluate(file_path, oracle=False, verbose=True, max_iter=4, critic=True):
    if verbose:
        print(file_path)

    em_scores = []
    f1_scores = []

    with open(file_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        for idx, sample in enumerate(data):
            if 'pred' not in sample:
                if "cot" in file_path:
                    sample['pred'] = [extract_cot_answer(sample['prediction']['greedy']['text'])]
                elif "direct" in file_path:
                    if 'greedy' not in sample['prediction']: # jump failed direct answer
                        continue
                    sample['pred'] = [sample['prediction']['greedy']['text']]

            cur_em = []
            cur_f1 = []

            for itr in range(max_iter):

                # stopped
                if itr > len(sample['pred']) - 1:
                    cur_em.append(cur_em[-1])
                    cur_f1.append(cur_f1[-1])
                    continue

                # the latest not NULL pred
                if critic:
                    for j in range(itr, -1, -1):
                        if not is_null_answer(sample['pred'][j]):
                            break
                    pred = sample['pred'][j]
                else:
                    pred = sample['pred'][itr]

                em, f1 = multi_ref_score(pred, sample['answer'])
                cur_em.append(em)
                cur_f1.append(f1)

                # early stop
                stop = (sample['pred'][itr] == sample['pred'][itr - 1])

                if (oracle and em) or (not oracle and stop):
                    cur_em.extend([em] * (max_iter - itr - 1))
                    cur_f1.extend([f1] * (max_iter - itr - 1))
                    break

            em_scores.append(cur_em)
            f1_scores.append(cur_f1)

    # output mean of each column of scores
    em_means = np.array(em_scores).mean(axis=0)
    em_means = list(np.round(em_means* 100, decimals=1))

    f1_means = np.array(f1_scores).mean(axis=0)
    f1_means = list(np.round(f1_means* 100, decimals=1))

    if verbose:
        print("num of samples:", len(em_scores))
        print(em_means)
        print(f1_means)
        print(f"CoT EM/F1:\t{em_means[0]} & {f1_means[0]}")
        print("CRITIC (oracle)" if oracle else "CRITIC", end=" ")
        print(f"EM/F1:\t{em_means[-1]} & {f1_means[-1]}\n")

    return em_means, f1_means


if __name__ == "__main__":

    ## text-davinci-003
    # critic
    # file_path = "outputs/text-davinci-003/ambig_qa/validation_critic_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/trivia_qa/validation_critic_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/hotpot_qa/validation_critic_500_seed0.jsonl"

    # critic no tools
    # file_path = "outputs/text-davinci-003/ambig_qa/validation_critic_no-tool_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/trivia_qa/validation_critic_no-tool_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/hotpot_qa/validation_critic_no-tool_500_seed0.jsonl"

    # direct
    # file_path = "outputs/text-davinci-003/ambig_qa/validation_direct_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/trivia_qa/validation_direct_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/hotpot_qa/validation_direct_500_seed0.jsonl"

    ## gpt-3.5-turbo
    # critic
    # file_path = "outputs/gpt-3.5-turbo/ambig_qa/validation_critic_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/trivia_qa/validation_critic_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/hotpot_qa/validation_critic_500_seed0.jsonl"

    # critic no tools
    # file_path = "outputs/gpt-3.5-turbo/ambig_qa/validation_critic_no-tool_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/trivia_qa/validation_critic_no-tool_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/hotpot_qa/validation_critic_no-tool_500_seed0.jsonl"

    # direct
    # file_path = "outputs/gpt-3.5-turbo/ambig_qa/validation_direct_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/trivia_qa/validation_direct_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/hotpot_qa/validation_direct_500_seed0.jsonl"

    # evaluate(file_path)
    # evaluate(file_path, oracle=True)
    # exit()

    ## react
    # for file_path in [
    #     ## gpt-3.5-turbo
    #     "outputs/gpt-3.5-turbo/ambig_qa/validation_react_500_seed0.jsonl",
    #     "outputs/gpt-3.5-turbo/trivia_qa/validation_react_500_seed0.jsonl",
    #     "outputs/gpt-3.5-turbo/hotpot_qa/validation_react_500_seed0.jsonl",

    #     ## text-davinci-003
    #     "outputs/text-davinci-003/ambig_qa/validation_react_500_seed0.jsonl",
    #     "outputs/text-davinci-003/trivia_qa/validation_react_500_seed0.jsonl",
    #     "outputs/text-davinci-003/hotpot_qa/validation_react_500_seed0.jsonl",
    # ]:
    #     evaluate(file_path, max_iter=2, critic=False)
    # exit()


    ## rejection sampling: best-of-n
    # for file_path in [
    #     "outputs/text-davinci-003/ambig_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/text-davinci-003/trivia_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/text-davinci-003/hotpot_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/gpt-3.5-turbo/ambig_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/gpt-3.5-turbo/trivia_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/gpt-3.5-turbo/hotpot_qa/validation_cot_500_seed0_t0.5.jsonl",
    # ]:
    #     print(file_path)
    #     rejection_sampling_eval(file_path, n=4)
    #     print()
    # exit()
    
    # all in one
    metrics = {}
    for data in ['ambig_qa', 'trivia_qa', "hotpot_qa"]:
        metrics[data] = {}
        for model in ['gpt-3.5-turbo', 'text-davinci-003']:
            metrics[data][model] = {}

            # critic
            #file_path = f"outputs/{model}/{data}/validation_critic_500_seed0.jsonl"
            with open('results_filename.txt', 'r', encoding='utf-8') as f:
                file_path = f.read()
            #file_path = "results/qa_critic/hotpot_qa/Meta-Llama-3.1-70B-Instruct-Turbo_results.json"
            em, f1 = evaluate(file_path, verbose=False)

            # # critic(oracle)
            # em_oracle, f1_oracle = evaluate(file_path, oracle=True, verbose=False)

            # # critic w/o tool
            # file_path = f"outputs/{model}/{data}/validation_critic_no-tool_500_seed0.jsonl"
            # em_notool, f1_notool = evaluate(file_path, verbose=False)

            # # react
            # file_path = f"outputs/{model}/{data}/validation_react_500_seed0_t0.jsonl"
            # em_react , f1_react = evaluate(file_path, max_iter=2, critic=False, verbose=False)
            
            # save em and f1 to metrics
            metrics[data][model]['em'] = em
            # metrics[data][model]['em_oracle'] = em_oracle
            # metrics[data][model]['em_notool'] = em_notool
            # metrics[data][model]['em_react'] = em_react
            metrics[data][model]['f1'] = f1
            # metrics[data][model]['f1_oracle'] = f1_oracle
            # metrics[data][model]['f1_notool'] = f1_notool
            # metrics[data][model]['f1_react'] = f1_react

    pprint.pprint(metrics, width=160)
