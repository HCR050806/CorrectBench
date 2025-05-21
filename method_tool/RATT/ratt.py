import re
import multiprocessing
from multiprocessing import Process, Queue
from langchain.tools import Tool
# from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from datetime import datetime
from difflib import unified_diff
from IPython.display import display, HTML
import argparse
import tiktoken
import sys
sys.path.append('/Self-Correction-Benchmark')
class RATT:
    def __init__(self, model, task=None, num_agents=1, num_steps=3, final_output_mode='only_last_step',
                 prompting_style='zero-shot-cot'):
        self.model = model
        self.task = task
        self.prompting_style = prompting_style
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.final_output_mode = final_output_mode
        newline_char = '\n'

    def num_tokens_from_string(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))

    def chunk_text_by_sentence(self, text, chunk_size=2048):
        """Chunk the $text into sentences with less than 2k tokens."""
        sentences = text.split('. ')
        chunked_text = []
        curr_chunk = []

        for sentence in sentences:
            if self.num_tokens_from_string(". ".join(curr_chunk)) + self.num_tokens_from_string(
                    sentence) + 2 <= chunk_size:
                curr_chunk.append(sentence)
            else:
                chunked_text.append(". ".join(curr_chunk))
                curr_chunk = [sentence]

        if curr_chunk:
            chunked_text.append(". ".join(curr_chunk))
        return chunked_text[0]

    def chunk_text_front(self, text, chunk_size=2048):
        '''
        get the first `trunk_size` token of text
        '''
        chunked_text = ""
        tokens = self.num_tokens_from_string(text)
        if tokens < chunk_size:
            return text
        else:
            ratio = float(chunk_size) / tokens
            char_num = int(len(text) * ratio)
            return text[:char_num]

    def chunk_texts(self, text, chunk_size=2048):
        '''
        trunk the text into n parts, return a list of text
        [text, text, text]
        '''
        tokens = self.num_tokens_from_string(text)
        if tokens < chunk_size:
            return [text]
        else:
            texts = []
            n = int(tokens / chunk_size) + 1
            part_length = len(text) // n
            extra = len(text) % n
            parts = []
            start = 0

            for i in range(n):
                end = start + part_length + (1 if i < extra else 0)
                parts.append(text[start:end])
                start = end
            return parts

    def split_draft(self, draft, split_char='\n\n'):
        draft_paragraphs = draft.split(split_char)
        return draft_paragraphs

    def get_query_wrapper(self, q, question, answer):
        result = self.get_query(question, answer)
        q.put(result)

    def get_query(self, question, answer):
        query_prompt = '''
            I want to verify the content correctness of the given question, especially the last sentences.
            Please summarize the content with the corresponding question.
            This summarization will be used as a query to search with Bing search engine.
            The query should be short but need to be specific to promise Bing can find related knowledge or pages.
            You can also use search syntax to make the query short and clear enough for the search engine to find relevant language data.
            Try to make the query as relevant as possible to the last few sentences in the content.
            **IMPORTANT**
            Just output the query directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
        '''
        query = self.model.query(f"##Question: {question}\n\n##Content: {answer}\n\n##Instruction: {query_prompt}")
        return query

    def get_content_wrapper(self, q, query):
        result = self.get_content(query)
        q.put(result)

    def get_content(self, query):
        res = self.get_search(query, 1)
        if not res:
            print(">>> No good Google Search Result was found")
            return None
        search_results = res[0]
        link = search_results['link']  # title, snippet
        res = self.get_page_content(link)
        if not res:
            print(f">>> No content was found in {link}")
            return None
        retrieved_text = res
        trunked_texts = self.chunk_texts(retrieved_text, 1500)
        trunked_texts = [trunked_text.replace('\n', " ") for trunked_text in trunked_texts]
        return trunked_texts

    def get_search(self, query: str = "", k: int = 1):  # get the top-k resources with google
        google_api_key = ""
        google_cse_id = ""
        search = GoogleSearchAPIWrapper(k=k, google_api_key=google_api_key, google_cse_id=google_cse_id)

        def search_results(query):
            return search.results(query, k)

        tool = Tool(
            name="Google Search Snippets",
            description="Search Google for recent results.",
            func=search_results,
        )
        ref_text = tool.run(query)
        if 'Result' not in ref_text[0].keys():
            return ref_text
        else:
            return None

    def get_page_content(self, link: str):
        loader = AsyncHtmlLoader([link])
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        if len(docs_transformed) > 0:
            return docs_transformed[0].page_content
        else:
            return None

    def get_revise_answer_wrapper(self, q, question, answer, content):
        result = self.get_revise_answer(question, answer, content)
        q.put(result)

    def get_revise_answer(self, question, answer, content):
        revise_prompt = '''
            I want to revise the answer according to retrieved related text of the question in WIKI pages.
            You need to check whether the answer is correct.
            If you find some errors in the answer, revise the answer to make it better.
            If you find some necessary details are ignored, add it to make the answer more plausible according to the related text.
            If you find that a part of the answer is correct and does not require any additional details, maintain that part of the answer unchanged. Directly output the original content of that part without any modifications.
            **IMPORTANT**
            Try to keep the structure (multiple paragraphs with its subtitles) in the revised answer and make it more structual for understanding.
            Split the paragraphs with `\n\n` characters.
            Just output the revised answer directly. DO NOT add additional explanations or annoucement in the revised answer unless you are asked to.
            If you have got the final answer, in the form \\boxed{answer}, at the end of your response.\n
        '''
        revised_answer = self.model.query(
            f"##Existing Text in Wiki Web: {content}\n\n##Question: {question}\n\n##Answer: {answer}\n\n##Instruction: {revise_prompt}")
        return revised_answer

    def RAG(self, question, draft_paragraphs):
        answer = ""
        for i, p in enumerate(draft_paragraphs):
            answer += '\n\n' + p
            # print(f"{i}: {answer}\n")
            res = self.run_with_timeout(self.get_query_wrapper, 3, question, answer)
            if not res:
                continue
            else:
                query = res

            res = self.run_with_timeout(self.get_content_wrapper, 5, query)
            if not res:
                continue
            else:
                content = res

            for j, c in enumerate(content):
                if j > 2:
                    break
                c_modified = self.filter_irrelevant_content(query, c)
                res = self.run_with_timeout(self.get_revise_answer_wrapper, 10, question, answer, c_modified)
                if not res:
                    continue
                else:
                    diff_html = self.generate_diff_html(answer, res)
                    display(HTML(diff_html))
                    answer = res
        return answer

    def filter_irrelevant_content(self, question, content):
        filter_prompt = '''
            Please read the following text and extract only the sections that are relevant to the given question. Organize the extracted information coherently, maintaining the structure of multiple paragraphs with subtitles, and split the paragraphs with `\n\n`.
            **Question**: {question}
            **Text to Filter**: {content}
            **Instruction**: Extract only the relevant information related to the question. Keep the structure clear with multiple paragraphs and subtitles. Provide the filtered information directly without additional explanations or commentary.
        '''
        filtered_content = self.model.query(
            f"##Text: {content}\n\n##Question: {question}\n\n##Instruction: Extract relevant information.")
        return filtered_content

    def get_draft_tot_inital(self, question, num_agents=3):
        draft_prompt = '''
            IMPORTANT:
            Try to answer this question/instruction with step-by-step thoughts and make the answer more structural.
            Use `\n\n` to split the answer into several paragraphs.
            Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
            If you have got the final answer, in the form \\boxed{answer}, at the end of your response.\n
            '''
        refine_prompt = '''
            Referencing the answers provided by all agents, synthesize a more detailed and comprehensive response by integrating all relevant details from these answers. Ensure logical coherence and provide ONLY THE MERGED ANSWER AS THE OUTPUT, omitting any discussion of the comparison process or analytical thoughts.
            If you have got the final answer, in the form \\boxed{answer}, at the end of your response.\n
            '''
        agents_drafts = []
        for i in range(num_agents):
            # print(f"hey!!!\n{question+draft_prompt}")
            draft = self.new_method(question, draft_prompt)
            draft_paragraphs = self.split_draft(draft)
            draft_modified = self.RAG(question, draft_paragraphs)
            agents_drafts.append(f"Agent{i + 1}: {draft_modified}")

        agents_input = '\n\n'.join(agents_drafts) + '\n\n' + refine_prompt
        final_draft = self.model.query(agents_input)
        return final_draft

    def new_method(self, question, draft_prompt):
        draft = self.model.query(question + draft_prompt)
        return draft

    def get_draft_tot(self, question, previous_answer, num_agents=3):
        draft_prompt = f'''
            Base your response on the provided question and the previous answer. Expand the answer by adding more details to enhance its comprehensiveness. Ensure that the expansion maintains logical coherence and enriches the details, making the response more thorough and well-structured.
            Question: {question}
            Previous Answer: {previous_answer}
            IMPORTANT:
            Answer the full question with step-by-step thoughts and make the answer more structural.
            Use `\n\n` to split the answer into several paragraphs.
            Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
        '''
        draft_prompt += "If you have got the final answer, in the form \\boxed{answer}, at the end of your response.\n"
        refine_prompt = '''
            Referencing the answers provided by all agents, synthesize a more detailed and comprehensive response by integrating all relevant details from these answers. Ensure logical coherence and provide ONLY THE MERGED ANSWER AS THE OUTPUT, omitting any discussion of the comparison process or analytical thoughts.
            If you have got the final answer, in the form \\boxed{answer}, at the end of your response.\n
        '''
        agents_drafts = []
        for i in range(num_agents):
            draft = self.model.query(draft_prompt)
            draft_paragraphs = self.split_draft(draft)
            draft_modified = self.RAG(question, draft_paragraphs)
            agents_drafts.append(f"Agent{i + 1}: {draft_modified}")

        agents_input = '\n\n'.join(agents_drafts) + '\n\n' + refine_prompt
        final_draft_raw = self.model.query(agents_input)
        revise_prompt = f'''
            Based on the original answer and an additional supplementary answer, generate a response that is richer in detail and logically coherent. Review the original answer:
            1. If any part of the answer is correct and requires no further details, retain that portion unchanged and output it directly as it is.
            2. For parts that may be improved or lack necessary details, enhance them by integrating information from the supplementary answer to make the response more comprehensive and accurate.
            3. If you identify any errors within the answers, correct these errors while ensuring that the revised content remains logically coherent.
            Original Answer: {previous_answer}
            Supplementary Answer: {final_draft_raw}

            **IMPORTANT**
            Ensure the revised answer maintains a structured format (multiple paragraphs with subtitles) for better clarity. Separate the paragraphs with `\n\n` characters. Output only the enhanced answer directly, without any extra explanations or announcements unless specifically requested.
        '''
        revise_prompt += "If you have got the final answer, in the form \\boxed{answer}, at the end of your response.\n"
        final_draft = self.model.query(revise_prompt)
        return final_draft

    def get_draft(self, question):
        draft_prompt = '''
            IMPORTANT:
            Try to answer this question/instruction with step-by-step thoughts and make the answer more structural.
            Use `\n\n` to split the answer into several paragraphs.
            Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
            If you have got the final answer, in the form \\boxed{answer}, at the end of your response.\n
        '''
        draft = self.model.query(f"{question}" + draft_prompt)
        return draft

    def run_with_timeout(self, func, timeout, *args, **kwargs):
        q = Queue()
        p = Process(target=func, args=(q, *args), kwargs=kwargs)
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()  # Terminate the process
            p.join()  # Ensure the process has been terminated
            result = None  # In case of a timeout, we do not have a result
        else:
            result = q.get()  # Retrieve the result from the queue
        return result

    def generate_diff_html(text1, text2):
        diff = unified_diff(text1.splitlines(keepends=True),
                            text2.splitlines(keepends=True),
                            fromfile='text1', tofile='text2')

        diff_html = ""
        for line in diff:
            if line.startswith('+'):
                diff_html += f"<div style='color:green;'>{line.rstrip()}</div>"
            elif line.startswith('-'):
                diff_html += f"<div style='color:red;'>{line.rstrip()}</div>"
            elif line.startswith('@'):
                diff_html += f"<div style='color:blue;'>{line.rstrip()}</div>"
            else:
                diff_html += f"{line.rstrip()}<br>"
        return diff_html

    def get_answer(self, output:str):
        answer = re.findall(r'\\boxed\{(.+?)\}', output)
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

    def __call__(self, question):
        draft = self.get_draft_tot_inital(question, self.num_agents)
        draft_paragraphs = self.split_draft(draft)
        answer_first_state = self.RAG(question, draft_paragraphs)
        previous_answer = answer_first_state
        each_step_drafts = [f"Step 1 \n: {previous_answer}"]

        for iteration in range(1, self.num_steps):
            draft = self.get_draft_tot(question, previous_answer, self.num_agents)
            draft_paragraphs = self.split_draft(draft)
            final_answer = self.RAG(question, draft_paragraphs)
            each_step_drafts.append(f"Step {iteration + 1} \n: {final_answer}")
            previous_answer = final_answer

        draft_cot = self.get_draft(question)
        if self.final_output_mode == 'combine_each_step':
            final_draft = '\n\n'.join(each_step_drafts)
            refine_prompt = f'''
                Referencing the answers provided by each step, synthesize a more detailed and comprehensive response by integrating all relevant details from these answers. Ensure logical coherence and provide ONLY THE MERGED ANSWER AS THE OUTPUT, omitting any discussion of the comparison process or analytical thoughts.
               '''
            refine_prompt += "If you have got the final answer, in the form \\boxed{answer}, at the end of your response.\n"
            previous_answer = self.model.query(final_draft + '\n\n' + refine_prompt)
        previous_answer = self.get_answer(previous_answer)
        return draft_cot, previous_answer


'''A test function for the class of RATT'''

import os
import json
import argparse
from tqdm import tqdm
import sys
from utils.process_config import open_config
from model import create_model
from task import create_task
from pathlib import Path

def test():
    # 添加路径
    sys.path.append('/home/zhaozeli/Self-Correction-Benchmark')

    # 创建解析器
    parser = argparse.ArgumentParser(description="Generate parameters for agent-based iterative output generation.")

    # 添加命令行参数
    parser.add_argument('--model_config', type=str,
                        default='/Self-Correction-Benchmark/config/model_config/api_gpt4o_config.json')
    parser.add_argument('--task_config_dir', type=str,
                        default='/Self-Correction-Benchmark/config/task_gpt4o_ratt')
    parser.add_argument('--method', type=str, default='ratt')
    parser.add_argument('--num_agents', default=1, type=int,
                        help='Number of agents used for generating outputs simultaneously.')
    parser.add_argument('--num_steps', default=3, type=int,
                        help='Number of iterative steps to run the generation process.')
    parser.add_argument('--final_output_mode', type=str, default='only_last_step',
                        choices=['combine_each_step', 'only_last_step'],
                        help='Method to generate the final output: "combine_each_step" to integrate outputs from each step, "only_last_step" to use the output from the final step as the final output.')

    args = parser.parse_args()

    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    # 获取 task_config 文件夹中的所有任务配置文件
    task_config_files = [f for f in Path(args.task_config_dir).glob('*.json')]
    if not task_config_files:
        print("No task config files found in the specified directory.")
        return

    # 遍历每个任务配置文件
    for task_config_path in task_config_files:
        task_config = open_config(config_path=str(task_config_path))
        task = create_task(task_config)
        data = task.get_data()

        num_agents = args.num_agents
        num_steps = args.num_steps
        final_output_mode = args.final_output_mode

        ratt = RATT(model, task, num_agents, num_steps, final_output_mode)

        results = []  # 用于保存结果
        correct_count = 0  # 计数正确答案的数量

        # 使用tqdm显示进度条
        for q, a in tqdm(zip(data['question'], data['final_answer']), total=len(data['question']), desc=f"Processing {task_config['task_name']}"):
            draft_cot, previous_answer = ratt(q)

            # 计算准确率：比较模型输出与真实答案
            if str(previous_answer) == a or str(previous_answer) == a[1:]:  # 假设完全匹配作为正确答案
                correct_count += 1
            print("------------------------------------")
            print(f"final_answer:{previous_answer}")
            print(f"correct_answer:{a}")
            print(str(previous_answer) == a or str(previous_answer) == a[1:])
            print("------------------------------------")
            result = {
                "question": q,
                "draft_cot": draft_cot,
                "final_answer": previous_answer,
                "correct_answer": a,
                "is_correct": str(previous_answer) == a or str(previous_answer) == a[1:]  # 是否正确
            }
            results.append(result)

            # 计算准确率（ACC）
            total_questions = len(data['question'])
            acc = correct_count / total_questions if total_questions > 0 else 0

            results_path = f'/Self-Correction-Benchmark/results_tool/{args.method}/{task.task_name}/'
            results_file = f'{results_path}/{model.name}_results.json'
            os.makedirs(os.path.dirname(results_file), exist_ok=True)

            # 将准确率放入文件的第一行，并保存结果到 JSON 文件
            final_output = {
                "ACC": acc,
                "results": results
            }

            # 保存结果到 JSON 文件
            with open(results_file, 'w') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)

            print(f"Results for {task_config['task_name']} have been saved to {results_file}")
            print(f"Accuracy (ACC): {acc * 100:.2f}%")

if __name__ == "__main__":
    test()
