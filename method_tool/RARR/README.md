# RARR Conda 环境设置与测试指南

## 环境激活

由于 RARR conda 环境已经设置好，您可以使用以下命令激活它：

```bash
conda activate RARR
```
## 设置环境变量

在运行测试之前，请设置 AZURE_SEARCH_KEY 环境变量。您可以通过以下命令设置：

```bash
export AZURE_SEARCH_KEY=your_subscription_key
```
将 your_subscription_key 替换为您的实际 Azure 订阅密钥。

## 运行测试

要运行测试，请执行以下命令：

```bash
python /Self-Correction-Benchmark/method_tool/RARR/rarr_test.py
```
## 重要说明

测试文件 rarr_test.py 会调用以下目录中的 Python 文件中的函数：

/Self-Correction-Benchmark/method_tool/RARR/prompts/ 下的 .py 文件

/Self-Correction-Benchmark/method_tool/RARR/utils/ 下的 .py 文件
