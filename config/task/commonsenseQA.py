import datasets
from .task_init import Task

class CommonsenseQATask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.data = datasets.load_dataset(self.data_path, "default")
        self.data = self.data[self.data_split]
        self.data = self.data[:self.data_num]
        self.get_answer()

    def get_answer(self):
        self.data['final_answer'] = []
        self.data['question0'] = self.data['question']
        self.data['choices'] = self.data['choices']

        self.data['question'] = [
            q + " " + " | ".join([f"{l}: {t}" for l, t in zip(c['label'], c['text'])])  # 拼接 label 和 text
            for q, c in zip(self.data['question0'], self.data['choices'])
        ]
        self.data['final_answer'] = [answerKey[0] for answerKey in self.data['answerKey']]

    def data_key_may(self):
        return {'question': 'question', 'answer': 'final_answer'}

    def get_data(self):
        return self.data
