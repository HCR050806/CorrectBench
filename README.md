<h1 align="center">
<br>
Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs
</h1>
<p align="center">
  <a href="https://github.com/zelizhao/CorrectBench.github.io"><b>[🌐 Website]</b></a> •
  <a href="https://github.com/zelizhao/CorrectBench"><b>[🐱 GitHub]</b></a>
  <br>
</p>
<p align="center">
This is a repository for building an unified framework and benchmark for Self-Correction of LLMs.

# 💡 Abstract
Self-correction of large language models (LLMs) emerges as a critical component for enhancing their reasoning performance. Although various self-correction methods have been proposed, a comprehensive evaluation of these methods remains largely unexplored, and the question of whether LLMs can truly correct themselves is a matter of significant interest and concern. In this study, we introduce \textbf{CorrectBench}, a benchmark developed to evaluate the effectiveness of self-correction strategies, including intrinsic, external, and fine-tuned approaches, across three tasks: commonsense reasoning, mathematical reasoning, and code generation. Our findings reveal that: 1) Self-correction methods can improve accuracy, especially for complex reasoning tasks; 2) Mixing different self-correction strategies yields further improvements, though it reduces efficiency; 3) Reasoning LLMs (e.g., DeepSeek-V3) have limited optimization under additional self-correction methods and have high time costs. Interestingly, a comparatively simple chain-of-thought (CoT) baseline demonstrates competitive accuracy and efficiency. These results underscore the potential of self-correction to enhance LLM's reasoning performance while highlighting the ongoing challenge of improving their efficiency. Consequently, we advocate for further research focused on optimizing the balance between reasoning capabilities and operational efficiency.
<p align="center">
    <img src="https://github.com/zelizhao/CorrectBench/blob/main/Self-Correction-Benchmark/overview.png" width="1000">
        <br>
    <em>An overview of the CorrectBench framework.</em>
</p>


# 📃Overview of this project
- **./config:** The config file of the models (`./config/model_config`), datasets and tasks (`./config/task_config`). 

- **./method:** Definition of different self-correction methods in our unified framework. We have support the following method: [RCI](https://arxiv.org/abs/2303.17491), [CoVe](https://arxiv.org/abs/2309.11495), [Reflexion](https://arxiv.org/abs/2303.11366), [Self-Refine](https://arxiv.org/abs/2303.17651).

- **./method_finetuning:** Definition of different self-correction methods in our unified framework. We have support the following method: [DCoT](https://arxiv.org/pdf/2407.03181).

- **./method_mixture:** We have a mixture of different self-correction methods

-  **./method_tool:** Definition of different self-correction methods in our unified framework. We have support the following method: [CRITIC](https://arxiv.org/abs/2305.11738), [RATT](https://arxiv.org/abs/2406.02746), [RARR](https://arxiv.org/abs/2210.08726).
  
- **./model:** Definition of loading API models and local models.

- **./task:** Definition of loading different datasets. We have support the following dataset: GSM8k, **TODO:** other datasets.

- **./utils:** Other assistive utilities.

# 🚀Preparation 
- **Environment settings:** `pip install -r ./requirement.txt`

- **Model settings:**   
Using API model of GPT series and Claude (see the [model list](https://api.keya.pw/pricing)), refer to the config file `./config/model_config/api_gpt_config.json`, set `"YOUR_API_KEY"` to your own API keys and `"model_method"` to `"api"`.   
Using other open-source API models from DeepInfra (see the [model list](https://deepinfra.com/models)), refer to the config file `./config/model_config/api_llama_config.json`, set `"YOUR_API_KEY"` to your own API keys and `"model_method"` to `"api"`.

# 😋Usage 
- **Usage Demo:** `./method/RCI.py` provides a demo for using [RCI](https://arxiv.org/abs/2303.17491) to solve dataset. Here is a demo for fast usage and other similar methods:   
```
parser = argparse.ArgumentParser(description="RCI Testing and Saving Script for Multiple Tasks")
parser.add_argument('--model_config', type=str, default='/Self-Correction-Benchmark/config/model_config/api_LLaMA3.1-70B.json',
                    help='Path to the model configuration file.')
parser.add_argument('--task_config_dir', type=str, default='/Self-Correction-Benchmark/config/task_config',
                    help='Path to the directory containing task configuration files.')
parser.add_argument('--method', type=str, default='rci',
                    help='Method name to use.')
parser.add_argument('--prompting_style', type=str, default='zero-shot-cot',
                    choices=['zero-shot-cot', 'few-shot-cot', 'zero-shot'],
                    help='Prompting style to use.')
parser.add_argument('--correct_iteration', type=int, default=1, 
                    help='Number of correction iterations.')
args = parser.parse_args()

test_and_save(args)

```

You can use `--model_config` to specify the model to use, and use `--task_config_dir` to specify the test data set, which is stored in the `results/{args.method}/{task.task_name}/'
results_file = f'{results_path}/{model.name}_results.json` folder by default. Here is an example:
```sh
python RCI.py \
    --model_config <Your Model Path> \
    --task_config_dir <Your Tasks Path> \
    --method rci\
    --prompting_style zero-shot-cot \
    --correct_iteration 1
```
Here's an example of what a JSON line in a **generation result file** might look like:
```json lines
{
"ACC": "ACC",
"empty_answers": "empty_answer_count",
"results": "final_results"
}
```
