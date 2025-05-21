import json

def extract_final_verified_responses(json_file_path):

    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 初始化一个空列表来存储 'Final Verified Response'
    final_verified_responses = []
    
    # 检查 'results' 键是否存在且是一个列表
    if 'results' in data and isinstance(data['results'], list):
        for item in data['results']:
            # 提取 'Final Verified Response' 并添加到列表中
            response = item.get('Final Verified Response')
            if response:
                final_verified_responses.append(response)
            else:
                # 如果某个条目中没有 'Final Verified Response'，可以选择记录或跳过
                final_verified_responses.append(None)  # 或者继续
    else:
        print("JSON 数据中没有 'results' 键或 'results' 不是一个列表。")
    
    return final_verified_responses

# 示例用法
if __name__ == "__main__":
    json_path = '/Self-Correction-Benchmark/method_mixture/Llama-3.1-8B-Instruct_results.json'  # 替换为你的 JSON 文件路径
    responses = extract_final_verified_responses(json_path)
    print(len(responses))
