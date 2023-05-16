import torch
import subprocess
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from utils.metrics import DifferenceEqualOpportunity, DifferenceAverageOdds

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024 * 1024  # bytes
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

    memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def get_best_device():
    if torch.cuda.is_available():
        best_device = None
        best_memory = 0
        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            available = get_gpu_memory()[i]
            if available > best_memory:
                best_memory = available
                best_device = device
        return best_device
    else:
        return torch.device("cpu")


# 初始化模型和分词器
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = get_best_device()
model.to(device)


# 定义一个线性分类器
# 假设任务是二分类任务
num_labels = 2
classifier = nn.Linear(model.config.hidden_size, num_labels).to(device)


# 定义一个MLP
mlp = nn.Sequential(
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
    nn.ReLU(),
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
).to(device)


# 初始化优化器
optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=2e-5)


# 用于计算cosine similarity的函数
cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)


# 设置epsilon
epsilon = 0.01


# 定义输入文本和标签
# 首先，将这些特征合并成一个文本字符串。
# 在以下示例中，我将 'Age', 'workclass', 'education', 'occupation', 'race', 'gender' 这些特征合并成一个文本字符串

def combine_features_raw(row):
    return f"{row['Age']} {row['workclass']} {row['education']} {row['occupation']} {row['race']} {row['gender']}"

def combine_features(row):
    return f"Age: {row['Age']} Workclass: {row['workclass']} Education: {row['education']} Occupation: {row['occupation']} Race: {row['race']} Gender: {row['gender']} "

class AdultDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']  # 请根据实际情况修改文本字段
        label = self.data.iloc[idx]['income']  # 请根据实际情况修改标签字段
        protected = self.data.iloc[idx]['Age']  # 请根据实际情况修改受保护特征字段
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'protected': torch.tensor(protected, dtype=torch.long)
        }

data = pd.read_csv("data/adult.tsv", sep='\t')
data = data.dropna()
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 请注意，BERT模型通常对自然语言文本的理解能力较强，如果特征本身足够表达它们的含义，那么在某些情况下，不包含列名可能也能取得良好的效果。
train_data['text'] = train_data.apply(combine_features, axis=1)
val_data['text'] = val_data.apply(combine_features, axis=1)

train_dataset = AdultDataset(train_data, tokenizer)
val_dataset = AdultDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# epochs = 5
epochs = 1
# 开始训练模型
for epoch in range(epochs):
    total_loss = 0
    total_correct = 0
    total_examples = 0
    total_DAO = 0
    total_DEO = 0

    # 开始训练模式
    model.train()
    classifier.train()

    for i, batch in enumerate(train_loader):
        inputs = {name: tensor.to(model.device) for name, tensor in batch.items() if name in ['input_ids', 'attention_mask']}
        labels = batch['label'].to(model.device)
        protected = batch['protected'].to(model.device)

        # 计算模型的输出
        outputs = model(**inputs)[0][:,0,:]

        # 计算原始的分类器损失
        logits = classifier(outputs)
        original_loss = nn.functional.cross_entropy(logits, labels)

        # 计算损失函数关于输入的梯度
        model.embeddings.word_embeddings.weight.requires_grad = True
        original_loss.backward(retain_graph=True)

        # 应用FGSM
        sign_gradient = model.embeddings.word_embeddings.weight.grad.data.sign()
        perturbed_embeddings = model.embeddings.word_embeddings.weight + epsilon * sign_gradient

        # 将对抗样本输入模型
        model.embeddings.word_embeddings.weight.data = perturbed_embeddings
        perturbed_outputs = model(**inputs)[0][:,0,:]

        # 计算扰动后的分类器损失
        perturbed_logits = classifier(perturbed_outputs)
        perturbed_loss = nn.functional.cross_entropy(perturbed_logits, labels)

        # 计算MLP损失
        mlp_outputs = mlp(outputs)
        mlp_perturbed_outputs = mlp(perturbed_outputs)
        mlp_similarity_loss = 1 - cosine_similarity(mlp_outputs, mlp_perturbed_outputs).mean()

        # 计算总损失
        total_batch_loss = original_loss + perturbed_loss + mlp_similarity_loss
        total_loss += total_batch_loss.item()

        # 反向传播和优化
        total_batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 计算准确率
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        total_correct += correct
        total_examples += labels.size(0)

        # 计算DAO和DEO
        y_pred = predicted
        y_real = {'income': labels, 'Age': batch['protected']}  # 假设'Age'是敏感特征
        privileged = 1
        unprivileged = 0

        batch_DEO = DifferenceEqualOpportunity(y_pred, y_real, 'Age', 'income', privileged, unprivileged, [0, 1])
        batch_DAO = DifferenceAverageOdds(y_pred, y_real, 'Age', 'income', privileged, unprivileged, [0, 1])

        total_DAO += batch_DAO
        total_DEO += batch_DEO

        # 打印每个批次的信息
        print(f"Epoch {epoch + 1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {total_batch_loss.item()}, Accuracy: {correct/labels.size(0)}, DAO: {batch_DAO}, DEO: {batch_DEO}")

    # 计算并打印平均损失
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss}")

    # 计算并打印准确率
    avg_accuracy = total_correct / total_examples
    print(f"Epoch {epoch + 1}/{epochs}, Avg Accuracy: {avg_accuracy}")

    # 计算并打印DAO和DEO
    avg_DAO = total_DAO / len(train_loader)
    avg_DEO = total_DEO / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Avg DAO: {avg_DAO}, Avg DEO: {avg_DEO}")

    print(y_pred)
    print(y_real)
