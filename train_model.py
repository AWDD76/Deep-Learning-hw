from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
from prepare_dataset import get_datasets
import torch

# 加载 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 分词预处理函数
def preprocess_function(example):
    return tokenizer(example["text"], truncation=True)

# 评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# 加载数据集
train_dataset, _ = get_datasets("train.jsonl", "test.jsonl")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_train = tokenized_train.rename_column("label", "labels")

# 创建训练参数
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_total_limit=1,
    logging_steps=10,
    no_cuda=False,           # 允许使用 GPU
)

# 构造 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# 训练模型
if __name__ == "__main__":
    trainer.train()

    # 保存模型和 tokenizer
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
