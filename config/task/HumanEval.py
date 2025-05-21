import datasets
from .task_init import Task

class HumanEvalTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.data = datasets.load_dataset(self.data_path, "openai_humaneval")
        self.data = self.data[self.data_split]
        self.data = self.data[:self.data_num]
        self.get_answer()

    def get_answer(self):
        self.data['question'] = []
        self.data['final_answer'] = []
        self.data['question'] = self.data['prompt']
        self.data['final_answer'] = self.data['canonical_solution']

    def data_key_may(self):
        return {'question': 'question', 'answer': 'final_answer'}

    def get_data(self):
        return self.data