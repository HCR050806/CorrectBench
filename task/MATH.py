import datasets
import re
from .task_init import Task

class MATHTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.data = datasets.load_dataset('json',data_dir=self.data_path)
        self.data = self.data[self.data_split]
        self.data = self.data[:self.data_num]
        self.get_answer()

    def get_answer(self):
        split_str = "\n####"
        self.data['question'] = []
        self.data['final_answer'] = []
        self.data['question'] = self.data['problem']
        answer = [solution for solution in self.data['solution']]
        # 正则表达式提取 \boxed{} 中的内容
        answers = []
        for item in answer:
            matches = re.findall(r"\\boxed\{(.*?)\}", item)
            answers.extend(matches)
        

    # 打印提取的结果


        self.data['final_answer'] = answers

    def data_key_may(self):
        return {'question': 'question', 'answer': 'final_answer'}

    def get_data(self):
        return self.data
