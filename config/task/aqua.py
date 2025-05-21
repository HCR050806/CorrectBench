import datasets
from .task_init import Task

class AQUATask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.data = datasets.load_dataset(self.data_path, "default")
        self.data = self.data[self.data_split]
        self.data = self.data[:self.data_num]
        self.get_answer()

    def get_answer(self):
        split_str = "####"
        self.data['question'] = []
        self.data['final_answer'] = []
        self.data['final_answer'] = [answer.split(split_str)[1] for answer in self.data['text']]
        split_str = "\nSOLUTION:"
        self.data['question'] = [answer.split(split_str)[0] for answer in self.data['text']]

    def data_key_may(self):
        return {'question': 'question', 'answer': 'final_answer'}

    def get_data(self):
        return self.data