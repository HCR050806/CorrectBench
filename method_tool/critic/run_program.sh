#!/bin/bash

# 运行第一个 Python 文件
echo "Running program_inference script..."
python program_inference.py --start_task 0 --end_task 100

echo "The inference results file is in /Self-Correction-Benchmark/method_tool/critic/results/program_inference."

# 运行第二个 Python 文件
echo "Running program_critic script..."
python program_critic.py --start_task 0 --end_task 100


echo "All scripts have been executed."
echo "The results file is in /Self-Correction-Benchmark/method_tool/critic/results/program_critic."