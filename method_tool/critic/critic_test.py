import json
import os
import random
import sys
import torch
import argparse
from tqdm import tqdm

sys.path.append('/Self-Correction-Benchmark')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
            prompt_path = "/Self-Correction-Benchmark/dataset/HotPotQA/inference_prompt.md"
            with open(prompt_path, 'r', encoding='utf-8') as fp:
                demo_prompt = fp.read().strip() + "\n\n"
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
            if temperature == 0:  # greedy answer
                res = self.model.query(prompt)
                prediction["greedy"]["text"] = res.strip()
                assert prediction['greedy']['text'] != "", "Empty answer"
            else:  # sampling
                res = self.model.query(prompt)
                for item in res['choices']:
                    prediction[f"temperature_{temperature}"]["text"].append(item['text'].strip())
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

        context = f"Q: {sample['question'].strip()}\nA: " 
        prediction = self.call_api(self.initial_prompt + context, num_sampling=self.num_sampling, temperature=self.temperature)
        sample['prediction'] = prediction
        return sample

def test_and_save(args):
    model = load_model(args)  # Assuming load_model is implemented elsewhere
    task, data = load_data(args)  # Assuming load_data is implemented elsewhere
    method = load_method(args)  # Assuming load_method is implemented elsewhere

    '''Create a directory to store the results'''
    results_path = f'./results/{args.method}/{args.task_config}/'
    results_file = f'{results_path}/{args.model_config}_results.json'
    dic = os.path.dirname(results_file)
    if not os.path.exists(dic):
        os.makedirs(dic)
    
    # Initialize a dictionary for results
    final_results = []

    # Initialize counters for accuracy
    correct_number = 0
    total_number = 0

    # Inference and evaluation
    for i in tqdm(range(len(data['question']))):
        total_number += 1
        sample = {
            'question': data['question'][i],
            'answer': data['final_answer'][i]  # Assuming 'final_answer' contains the ground truth
        }
        
        # Run inference
        inference_method = QA_Inference(model, task)
        record = inference_method(sample)

        # Evaluate correctness (assuming a simple string comparison for this example)
        correct = evaluate_answer(record['prediction'], data['final_answer'][i])  # Assuming this function exists
        record['correct'] = correct

        # Count correct answers
        if correct:
            correct_number += 1
        
        final_results.append(record)

    # Calculate accuracy
    ACC = correct_number / total_number
    print(f"Method: {args.method}\nTask: {task.task_name}\nModel: {model.name}\n Accuracy: {ACC:.2f}")

    '''Save the results to a JSON file, with ACC in the first line'''
    results = {
        'accuracy': ACC,
        'results': final_results
    }

    # Write to the file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")

def evaluate_answer(prediction, ground_truth):
    # A simple evaluation function (you can adjust this logic based on your specific needs)
    predicted_answer = prediction.get('greedy', {}).get('text', '').strip()
    return predicted_answer == ground_truth  # Exact match evaluation

def load_model(args):
    # Dummy function for model loading
    # Replace with actual model loading logic
    return torch

def load_data(args):
    # Dummy function for data loading
    # Replace with actual data loading logic
    return {}, [{'question': 'Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?',
                 'final_answer': 'no'}]

def load_method(args):
    # Dummy method loader
    # Replace with actual method logic
    return QA_Inference

def parse_args():
    parser = argparse.ArgumentParser(description="Test and Save Inference Results with Accuracy")
    
    # Model and task configurations
    parser.add_argument('--method', type=str, default='critic')
    parser.add_argument('--task_config', type=str, default='/Self-Correction-Benchmark/config/task_config/HotpotQA.json')
    parser.add_argument('--model_config', type=str, default='/Self-Correction-Benchmark/config/model_config/api_LLaMA3.1-70B.json')
    
    # Optional parameters
    parser.add_argument('--start_task', type=int, default=0, help='Start index for tasks')
    parser.add_argument('--end_task', type=int, default=100, help='End index for tasks')
    
    # Data and result paths
    parser.add_argument('--data_file', type=str, default='/Self-Correction-Benchmark/dataset/HotPotQA/validation.jsonl')
    
    return parser.parse_args()

# To run the test with argparse
if __name__ == '__main__':
    args = parse_args()
    test_and_save(args)
