
class Task:
    def __init__(self, config):
        self.data_path = config["data_path"]
        self.data_split = config["data_split"]
        self.data_num = config["data_num"]
        self.task_name = config["task_name"]
        self.task_type = config["task_type"]
        self.print_task_info()
    
    def print_task_info(self):
        print(f"{'-'*len(f'| Task name: {self.task_name}')}\n| Task name: {self.task_name}\n| Task type: {self.task_type}\n| Data split: {self.data_split}\n| Data number {self.data_num}\n{'-'*len(f'| Task name: {self.task_name}')}")

