import re
import gym
import string
from typing import Tuple
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
class QAEnv(gym.Env):
    def __init__(self,
                 model,
                #  question: str,
                #  key: str,
                 max_steps: int = 6,
                 explorer: DocstoreExplorer = DocstoreExplorer(Wikipedia())):
        
        self.question = ''
        self.key = ''
        self.model = model
        self.max_steps = max_steps
        self.explorer = explorer

        self.reset()

    def reset(self):
          self.curr_step = 1
          self.terminated = False
          self.answer = ''

    def step(self, action: str) -> Tuple[str, bool, bool, bool, bool]:
        action_type, argument = parse_action(action)
        if(action_type!=None):
            action_type = action_type[0].upper() + action_type[1:]
        if action_type == 'Finish' and argument != None:
            self.answer = argument
            
            final_prompt=f"Q:{self.question}\n"+f"You are given the final step of a problem-solving process. Based on this step, provide **only the final answer** and enclose it in `{{}}`. Do not include any additional steps, explanations, or text. Do not perform any further reasoning or thinking—simply extract the answer directly from the final step and put it into the `{{}}`.\nYou mustn't think further!\n**Final Step:**" + self.answer + f"\n**Answer:** {{}}"
            print("\033[96m" + final_prompt + "\033[0m")
            text = self.model.query(final_prompt)
            print("\033[97m" + text + "\033[0m")
            pattern = r'\{(.*?)\}'
            matches = re.findall(pattern, text, re.DOTALL)
            if(len(matches)>0):
                self.answer = matches[-1]
                print("\033[89m" + text + "\033[0m")
            if self.is_correct():
                observation = 'Answer is CORRECT'
            else: 
                observation = 'Answer is INCORRECT'
            self.terminated = True

        elif action_type == 'Search':
            try:
                observation = self.explorer.search(argument).strip('\n').strip()
            except Exception as e:
                print(e)
                observation = f'Could not find that page, please try again.'
                    
        elif action_type == 'Lookup':
            try:
                observation = self.explorer.lookup(argument).strip('\n').strip()
            except ValueError:
                observation = f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'

        else:
            observation = 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'
        
        print("\033[91m" + observation + "\033[0m")
        reward = self.is_correct()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        self.curr_step += 1

        

        return observation, reward, terminated, truncated, self.curr_step

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)
    
    def is_terminated(self) -> bool:
        return self.terminated

    def is_truncated(self) -> bool:
        return self.curr_step >= self.max_steps
    
    def __call__(self, question, key):
        self.question = question
        self.key = key

def parse_action(string):
    print("\033[91m" + "*" * 50 + "\033[0m")
    print(string)
    # 找到 ] 的位置
    index = string.find(']')

    # 如果找到 ]，截取 ] 及其之前的内容
    if index != -1:
        string = string[:index + 1]
        
    pattern = r'^(.+?)\[(.+)\]$'
    # pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        action_type = action_type[-6:]
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None, None

def normalize_answer(s):
    print("\033[95m" + s + "\033[0m")
    def remove_articles(text):
        print("\033[95m" + s + "\033[0m")
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        print("\033[95m" + s + "\033[0m")
        return " ".join(text.split())   
    def remove_punc(text):
        print("\033[95m" + s + "\033[0m")
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)  
    def lower(text):
        print("\033[95m" + s + "\033[0m")
        return text.lower() 
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)

