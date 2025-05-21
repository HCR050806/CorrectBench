# Benchmark for Self-Correction
This is a repository for building an unified framework and benchmark for Self-Correction of LLMs.

# 📃Overview of this project
- **./config:** The config file of the models (`./config/model_config`), datasets and tasks (`./config/task_config`). 

- **./method:** Definition of different self-correction methods in our unified framework. We have support the following method: [RCI](https://arxiv.org/abs/2303.17491), **TODO:** other methods.

- **./model:** Definition of loading API models and local models.

- **./task:** Definition of loading different datasets. We have support the following dataset: GSM8k, **TODO:** other datasets.

- **./utils:** Other assistive utilities.

# 🚀Preparation 
- **Environment settings:** `pip install -r ./requirement.txt`

- **Model settings:**   
Using API model of GPT series and Claude (see the [model list](https://api.keya.pw/pricing)), refer to the config file `./config/model_config/api_gpt_config.json`, set `"YOUR_API_KEY"` to your own API keys and `"model_method"` to `"api"`.   
Using other open-source API models from TogetherAI (see the [model list](https://api.together.ai/models)), refer to the config file `./config/model_config/api_llama_config.json`, set `"YOUR_API_KEY"` to your own API keys and `"model_method"` to `"api"`.  
Using local model with **Transformer**, refer to the config file `./config/model_config/llama_config.json`, set `"name"` to the Huggingface model name or you local model path and `"model_method"` to `"local"`.

# 😋Usage 
- **Usage Demo:** `demo.py` provides a demo for using [RCI](https://arxiv.org/abs/2303.17491) to solve the GSM8k dataset. Here is a demo for fast usage:   
```
from model import create_model
from method import create_method
from utils.process_config import open_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='rci')
parser.add_argument('--prompting_style', type=str, default='zero-shot-cot')
args = parser.parse_args()

model_config = open_config(config_path='./config/model_config/api_llama_config.json')
model = create_model(model_config)

correction_method = create_method(args, model)
question = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
answer = "3"

results = correction_method(question, answer)
print(result)

```
