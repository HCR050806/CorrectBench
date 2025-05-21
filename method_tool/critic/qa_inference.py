import re
import os
import json
import func_timeout
from typing import Union, Any
from math import isclose
import random
import sys
sys.path.append('/Self-Correction-Benchmark')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
print("CUDA是否可用:", torch.cuda.is_available())
print("可用的GPU数量:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
#
# '''{"id":"5adbf0a255429947ff17385a",
# "question":"Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
# "answer":"no",
# "type":"comparison",
# "level":"hard",
# "supporting_facts":{"title":["Laleli Mosque","Esma Sultan Mansion"],"sent_id":[0,0]},
# "context":{"title":["Esma Sultan (daughter of Abd\u00fclaziz)","Djama\u00e2 el Kebir","K\u00fc\u00e7\u00fck H\u00fcseyin Pasha","Esma Sultan (daughter of Abdul Hamid I)","Sultan Ahmed Mosque","Laleli Mosque","Esma Sultan Mansion","Esma Sultan","Gevheri Kad\u0131n","Esma Sultan (daughter of Ahmed III)"],
# "sentences":[["Esma Sultan (21 March 1873 \u2013 7 May 1899) was an Ottoman princess, the daughter of Sultan Abd\u00fclaziz and his wife Gevheri Kad\u0131n, herself the daughter of Salih Bey Svatnba."," She was the half-sister of Abd\u00fclmecid II, the last Caliph of the Muslim world."],["The Great Mosque of Algiers (Arabic: \u0627\u0644\u062c\u0627\u0645\u0639 \u0627\u0644\u0643\u0628\u064a\u0631\u200e \u200e , \"Jemaa Kebir\") or \u201cDjama\u2019a al-Kebir\u201d (meaning Great Mosque) is a mosque in Algiers, Algeria, located very close to Algiers Harbor."," An inscription on the minbar (\u0645\u0646\u0628\u0631) or the pulpit testifies to fact that the mosque was built in 1097."," It is also known by several other names such as Grand Mosque d'Alger, Djamaa al-Kebir, El Kebir Mosque and Jami Masjid."," It is one of the few remaining examples of Almoravid architecture."," It is the oldest mosque in Algiers and is said to be the oldest mosque in Algeria after Sidi Okba Mosque."," It was built under sultan Ali ibn Yusuf."," Its minaret dates from 1332 (1324 in some sources) and was built by the Ziyyanid Sultan of Tlemcen."," The gallery at the outside of the mosque was built in 1840."," Its construction was a consequence of a complete reconstruction of the street by the French."],["K\u00fc\u00e7\u00fck H\u00fcseyin Pasha (1757 \u2013 7 December 1803), also known as Tayazade Damat K\u00fc\u00e7\u00fck H\u00fcseyin Pasha, was an Ottoman statesman and admiral who was Kapudan Pasha (Grand Admiral of the Ottoman Navy) from 11 March 1792 to 7 December 1803."," He was a \"damat\" (\"bridegroom\") to the Ottoman dynasty after he married an Ottoman princess, Esma Sultan."],["Esma Sultan (17 July 1778 \u2013 4 June 1848) was an Ottoman princess, daughter of Sultan Abdul Hamid I, sister of Sultan Mustafa IV and Sultan Mahmud II."," She was the adoptive mother of Bezmi\u00e2lem Sultan and Rahime Perestu Sultan."],["The Sultan Ahmed Mosque or Sultan Ahmet Mosque (Turkish: \"Sultan Ahmet Camii\" ) is a historic mosque located in Istanbul, Turkey."," A popular tourist site, the Sultan Ahmed Mosque continues to function as a mosque today; men still kneel in prayer on the mosque's lush red carpet after the call to prayer."," The Blue Mosque, as it is popularly known, was constructed between 1609 and 1616 during the rule of Ahmed I."," Its K\u00fclliye contains Ahmed's tomb, a madrasah and a hospice."," Hand-painted blue tiles adorn the mosque\u2019s interior walls, and at night the mosque is bathed in blue as lights frame the mosque\u2019s five main domes, six minarets and eight secondary domes."," It sits next to the Hagia Sophia, another popular tourist site."],["The Laleli Mosque (Turkish: \"Laleli Camii, or Tulip Mosque\" ) is an 18th-century Ottoman imperial mosque located in Laleli, Fatih, Istanbul, Turkey."],["The Esma Sultan Mansion (Turkish: \"Esma Sultan Yal\u0131s\u0131\" ), a historical yal\u0131 (English: waterside mansion ) located at Bosphorus in Ortak\u00f6y neighborhood of Istanbul, Turkey and named after its original owner Esma Sultan, is used today as a cultural center after being redeveloped."],["Esma Sultan is the name of three daughters of three Ottoman Sultans:"],["Gevheri Kad\u0131n (8 July 1856\u00a0\u2013 6 September 1884) was the fifth wife of 32nd Ottoman Sultan Abd\u00fclaziz."," She was the mother of \u015eehzade Mehmed Seyfeddin and Esma Sultan of the Ottoman Empire."],["Esma Sultan (14 March 1726 \u2013 13 August 1788) was an Ottoman princess, daughter of Sultan Ahmed III and his consort Zeynep Kad\u0131n.",
# " She was the half-sister of Sultan Mustafa III and Abdul Hamid I."]]}}
# '''
#
class QA_Inference:
    def __init__(self, model, task=None, prompting_style='zero-shot-cot', correct_iteration=1):
        self.model = model
        self.task = task
        self.prompting_style = prompting_style
        self.initial_prompt()
        self.cririque_promtp = 'Review your previous answer and find problems with your answer.\n\n'
        self.improve_prompt = 'Based on the problems you found, improve your answer. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{answer}.\n\n'
        self.correct_iteration = correct_iteration
        self.temperature = 0
        self.num_sampling = 1

    def initial_prompt(self):
        if self.prompting_style == 'zero-shot-cot':
            self.initial_prompt = "Let's think step by step. Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.\n\nA:"
        elif self.prompting_style == 'few-shot-cot':
            #self.initial_prompt = "A:\n"   #TODO: add the few-shot prompt file
            prompt_path = "/Self-Correction-Benchmark/dataset/HotPotQA/inference_prompt.md"
            with open(prompt_path, 'r', encoding='utf-8') as fp:
                demo_prompt = fp.read().strip() + "\n\n"
            #full_prompt = demo_prompt + f'Question: {question}' + '\n'
            self.initial_prompt = demo_prompt + '\n'
        elif self.prompting_style == 'zero-shot':
            self.initial_prompt = "Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.\nA:\n"
        else:
            print("WARNING: The prompting style is not given. Use zero-shot-cot as default.")
            self.initial_prompt = "Let's think step by step. Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.\nA:\n"

    def call_api(self, prompt, num_sampling, verbose=True, temperature=0):
        if temperature == 0:
            prediction = {"greedy": {}}
        else:
            prediction = {}
            prediction[f'temperature_{temperature}'] = {"text": [], "logprobs": [], "tokens": []}

        try:
            if temperature == 0: # greedy answer
                #res = llm(prompt, model, stop=["\n\n"], logprobs=1)['choices'][0]
                #res = llm(prompt, model, stop=["\n\n"], logprobs=1).choices[0]
                res = self.model.query(prompt)
                print("res",res)
                #prediction["greedy"]["text"] = res['text'].strip()
                prediction["greedy"]["text"] = res.strip()
                assert prediction['greedy']['text'] != "", "Empty answer"
                # tokens & logprobs
                # end_idx = get_end_index(res['logprobs']['tokens'])
                # prediction["greedy"]["tokens"] = res['logprobs']['tokens'][:end_idx]
                # prediction["greedy"]["logprobs"] = res['logprobs']['token_logprobs'][:end_idx]

            else: # sampling
                #res = llm(prompt, model, stop=["\n\n"], temperature=temperature, n=num_sampling, logprobs=1)
                res = self.model.query(prompt)
                for item in res['choices']:
                    prediction[f"temperature_{temperature}"]["text"].append(item['text'].strip())
                    # tokens & logprobs
                    # end_idx = get_end_index(item['logprobs']['tokens'])
                    # tokens = item['logprobs']['tokens'][:end_idx]
                    # token_logprobs = item['logprobs']['token_logprobs'][:end_idx]
                    # prediction[f"temperature_{temperature}"]["tokens"].append(tokens)
                    # prediction[f"temperature_{temperature}"]["logprobs"].append(token_logprobs)
            return prediction
        except:
            return {}

    def correct(self, initial_input, output):
        critique_input = initial_input + output + '\n\n' + self.cririque_promtp
        critique_output = self.model.query(critique_input)
        improve_input = critique_input + '\n\n' + critique_output + '\n\n' + self.improve_prompt
        improve_output = self.model.query(improve_input)
        return critique_output, improve_output
        
    def __call__(self, sample):
        entries_to_remove = ["context", "used_queries", "nq_doc_title"]
        for key in entries_to_remove:
            if key in sample:
                sample.pop(key, None)

        # process question & answer
        if self.task.task_name == "ambig_qa":
            if sample['annotations']['type'][0] == "singleAnswer":
                # single answer
                answers = sample['nq_answer']
                for ans in sample['annotations']['answer']:
                    answers.extend(ans)
                sample['answer'] = list(set(answers))
            else:
                # random choose a question with multiple answers
                qa_pairs = sample['annotations']['qaPairs'][0]
                rand_i = random.randint(0, len(qa_pairs['question'])-1)
                sample['question'] = qa_pairs['question'][rand_i]
                sample['answer'] = qa_pairs['answer'][rand_i]

        context = f"Q: {sample['question'].strip()}\nA: " 

        #print(f"idx: {idx}")
        print(context, end="")

        prediction = self.call_api( self.initial_prompt + context, num_sampling=self.num_sampling, temperature=self.temperature)

        sample['prediction'] = prediction
        print(prediction)

        if 'greedy' in prediction:
            print(prediction['greedy']['text'])
        print()

        return sample
        



'''A test function for the class of RCI'''
def test():
    import sys
    #sys.path.append('/mnt/yuanzenghui/Self-Correction-Benchmark')
    #sys.path.append('/Self-Correction-Benchmark')
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
    parser.add_argument('--model_config', type=str, default='/Self-Correction-Benchmark/config/model_config/api_LLaMA3.1-70B.json')
    parser.add_argument('--task_config', type=str, default='/Self-Correction-Benchmark/config/task_config/HotpotQA.json')
    #parser.add_argument('--task_config', type=str, default='config/task_config/hotpot_qa.json')
    parser.add_argument('--method', type=str, default='qa_inference')
    parser.add_argument('--prompting_style', type=str, default='few-shot-cot')
    parser.add_argument('--data_file', type=str, default='/Self-Correction-Benchmark/dataset/HotPotQA/train.json')

    args = parser.parse_args()


    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    task_config = open_config(config_path=args.task_config)
    task = create_task(task_config)
    print(task.task_name)
    #return
    #data = task.get_data()

    '''Create a directory to store the results'''
    results_path = f'results/{args.method}/{task.task_name}/'
    global results_file
    results_file = f'{results_path}/{model.name}_results_{args.start_task}_{args.end_task}.json'
    with open('results_filename.txt', 'w', encoding='utf-8') as f:
        f.write(results_file)

    dic = os.path.dirname(results_file)
    if not os.path.exists(dic):
        os.makedirs(dic)
    with open(results_file, 'w') as f:
        json.dump({}, f)
    print(f"Make a new file {results_file} to save the inference result.")
    
    #data_file = f"../dataset/Hotpot_qa/validation.jsonl"
    dataset = []
    # load data
    with open(args.data_file, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    #print("dataset example: \n",dataset[0])        

    #inference
    inference_result = []
    idx = args.start_task
    for idx, sample in enumerate(dataset):
        if idx >= args.end_task:
            break
        if idx < args.start_task or (args.end_task != -1 and idx >= args.end_task):
            continue
        #将idx编号加进example里
        #example = {**{'idx': idx}, **example}
        inference_method = QA_Inference(model, task, args.prompting_style)
        record = inference_method(sample)
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
