import datasets
import random
import pandas as pd
from .task_init import Task

class GPQATask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.data = datasets.load_dataset('csv', data_files=self.data_path)
        self.data = self.data[self.data_split]
        self.data = self.data[:self.data_num]
        self.data = pd.DataFrame(self.data)
        self.get_answer()

    def get_answer(self):
        self.data['final_question'] = ''
        self.data['final_answer'] = ''
        for index,row in self.data.iterrows():
            question = row['Question']
            correct_answer = row['Correct Answer']
            incorrect_answers = [
                row['Incorrect Answer 1'],
                row['Incorrect Answer 2'],
                row['Incorrect Answer 3']
            ]
            options = [correct_answer] + incorrect_answers
            shuffled_options = list(enumerate(options))
            random.shuffle(shuffled_options)
            formatted_question = f"{question}\n\nOptions:\n"
            for i, option in shuffled_options:
                formatted_question += f"- {option}\n"
            self.data.at[index,'question'] = formatted_question     #Pay attention to the difference between Question and question
            self.data.at[index,'final_answer'] = row['Correct Answer']
        self.data = self.data.to_dict(orient='list')

    def data_key_may(self):
        return {'question': 'final_question', 'answer': 'final_answer'}

    def get_data(self):
        return self.data
