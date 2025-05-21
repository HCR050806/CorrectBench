#!/bin/bash

# 运行第一个 Python 文件
echo "Running qa_inference script..."
python qa_inference.py --start_task 0 --end_task 2

echo "The inference results file is in /Self-Correction-Benchmark/method_tool/critic/results/qa_inference."

# 运行第二个 Python 文件
echo "Running qa_critic script..."
python qa_critic.py --start_task 0 --end_task 2


echo "All scripts have been executed."
echo "The results file is in /Self-Correction-Benchmark/method_tool/critic/results/qa_critic."