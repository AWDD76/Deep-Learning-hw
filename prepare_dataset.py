from datasets import Dataset
from load_data import load_jsonl

def get_datasets(train_path: str, test_path: str):
    # 加载 JSONL 格式数据
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    # 转换为 Huggingface 的 Dataset 对象
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    return train_dataset, test_dataset
