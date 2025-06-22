from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import json
from tqdm import tqdm

# 选择 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
model_dir = "./results/checkpoint-5250"  # ✅ 根据你的训练输出
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_dir)
model.to(device)
model.eval()

# 加载 test.jsonl
test_texts = []
with open("test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        test_texts.append(data["text"])

# 分词与编码
encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
input_ids = encodings["input_ids"].to(device)
attention_mask = encodings["attention_mask"].to(device)

# 构造 DataLoader
dataset = TensorDataset(input_ids, attention_mask)
dataloader = DataLoader(dataset, batch_size=32)

# 执行预测
predictions = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting"):
        input_ids_batch, attention_mask_batch = batch
        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1)
        predictions.extend(batch_preds.cpu().tolist())

# 保存预测结果到 submit.txt
with open("submit.txt", "w", encoding="utf-8") as f:
    for label in predictions:
        f.write(str(label) + "\n")

print("✅ 预测完成，已生成 submit.txt！共计预测", len(predictions), "条样本。")
