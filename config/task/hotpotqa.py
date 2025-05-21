from .task_init import Task
import json
import pandas as pd

class HotPotQATask(Task):
    def __init__(self, config):
        super().__init__(config)
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.data = self.data[:self.data_num]
        self.data = pd.DataFrame(self.data)
        self.get_answer()

    def get_answer(self):
        self.data['final_answer'] = ''
        self.data['final_answer'] = self.data['answer']
        for index,row in self.data.iterrows():
            final_question = row['question']+'\n'+'\n'.join(context[0] + '\n' + ''.join(context[1]) for context in row['context'])
            self.data.at[index,'question'] = final_question
        self.data = self.data.to_dict(orient='list')

    def data_key_may(self):
        return {'question': 'question', 'answer': 'final_answer'}

    def get_data(self):
        return self.data  
