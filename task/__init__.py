from .gsm import GSMTask
from .aqua import AQUATask
from .gpqa import GPQATask
from .commonsenseQA import CommonsenseQATask
from .csqa import CSQATask
from .HumanEval import HumanEvalTask
from .MATH import MATHTask
from .hotpotqa import HotPotQATask

def create_task(config):
    """
    Factory method to create a task instance for self-correction
    """
    if config["task_name"].lower() == "gsm8k":
        return GSMTask(config)
    elif config["task_name"].lower() == "aqua":
        return AQUATask(config)
    elif config["task_name"].lower() == "gpqa":
        return GPQATask(config)
    elif config["task_name"].lower() == "commonsenseqa":
        return CommonsenseQATask(config)
    elif config["task_name"].lower() == "csqa":
        return CSQATask(config)
    elif config["task_name"].lower() == "humaneval":
        return HumanEvalTask(config)
    elif config["task_name"].lower() == "math":
        return MATHTask(config)
    elif config["task_name"].lower() == "csqa":
        return CSQATask(config)
    elif config["task_name"].lower() == "hotpotqa":
        return HotPotQATask(config)
    else:
        raise ValueError(f"ERROR: Unknown task {config['task_info']['task_name']}.")