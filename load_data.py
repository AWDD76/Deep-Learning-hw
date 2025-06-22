import json

def load_jsonl(path):
    """
    加载 .jsonl 文件，返回列表，每行为一个字典。
    """
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]
