import re
import os
import json
from tqdm import tqdm
import argparse
import glob
import sys
import torch
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
            return "A:\n"  # TODO: add the few-shot prompt file
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

    def get_answer(self, output):
        """
        Extracts the answer from the model's output. It looks for the pattern \boxed{answer}.
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

    def correct(self, initial_input, output):
        critique_input = initial_input + output + '\n\n' + self.critique_prompt
        critique_output = self.model.query(critique_input)
        improve_input = critique_input + '\n\n' + critique_output + '\n\n' + self.improve_prompt
        improve_output = self.model.query(improve_input)
        return critique_output, improve_output

    def __call__(self, question, answer):
        try:
            initial_input = 'Q: ' + question + '\n\n' + self.initial_prompt
            output = self.model.query(initial_input)
            record = {}
            record['round_0'] = output


            for iter_num in range(self.correct_iteration):
                critique, output = self.correct(initial_input, output)
                record[f'round_{iter_num + 1}'] = {'critique': critique, 'output': output}
                

                final_answer = self.get_answer(output)
                record['final_answer'] = final_answer
                record['correct answe'] = answer
                record['question'] = question
                print("-----------------------------------------")
                print(f"answer:{final_answer}")
                print(f"correct answer:{answer}")
                print("-----------------------------------------")


                if str(final_answer) == answer:
                    record['correct'] = True
                    break  
                elif str(final_answer) == answer[1:]: 
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



def test_and_save(args):
    from utils.process_config import open_config
    from model import create_model
    from task import create_task


    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    task_config_files = glob.glob(os.path.join(args.task_config_dir, '*.json'))

    if not task_config_files:
        print(f"No task configuration files found in {args.task_config_dir}.")
        return

    for task_config_path in task_config_files:
        task_name = os.path.splitext(os.path.basename(task_config_path))[0]
        print(f"\nProcessing Task: {task_name}")

        task_config = open_config(config_path=task_config_path)
        task = create_task(task_config)
        data = task.get_data()

        method = RCI(model, task, args.prompting_style, correct_iteration=args.correct_iteration)

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
            record = method(question, answer)
            final_results.append(record)
            if record.get('correct', False):
                correct_number += 1
            if record.get('final_answer') is None:
                empty_answer_count += 1

            ACC = correct_number / (i + 1 - empty_answer_count) if (i + 1 - empty_answer_count) > 0 else 0
            results_dict = {
                "ACC": ACC,
                "empty_answers": empty_answer_count,
                "results": final_results
            }

            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=4)

        print(f"Method: {args.method}\nTask: {task.task_name}\nModel: {model.name}\nFinal Accuracy: {ACC:.2f}")
        print(f"Number of questions with empty answers: {empty_answer_count}")

        print(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="RCI Testing and Saving Script for Multiple Tasks")
    parser.add_argument('--model_config', type=str, default='/Self-Correction-Benchmark/config/model_config/api_LLaMA3.1-70B.json',
                        help='Path to the model configuration file.')
    parser.add_argument('--task_config_dir', type=str, default='/Self-Correction-Benchmark/config/task_config',
                        help='Path to the directory containing task configuration files.')
    parser.add_argument('--method', type=str, default='rci',
                        help='Method name to use.')
    parser.add_argument('--prompting_style', type=str, default='zero-shot-cot',
                        choices=['zero-shot-cot', 'few-shot-cot', 'zero-shot'],
                        help='Prompting style to use.')
    parser.add_argument('--correct_iteration', type=int, default=1, 
                        help='Number of correction iterations.')
    args = parser.parse_args()

    test_and_save(args)


if __name__ == "__main__":
    main()
