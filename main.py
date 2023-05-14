import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# 初始化模型和分词器
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# 定义一个线性分类器
classifier = nn.Linear(model.config.hidden_size, 2).to(device)  # 假设任务是二分类任务


# 定义一个MLP
mlp = nn.Sequential(
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
    nn.ReLU(),
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
).to(device)


# 初始化优化器
optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)


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
train_data['text'] = train_data.apply(combine_features_raw, axis=1)
val_data['text'] = val_data.apply(combine_features_raw, axis=1)

train_dataset = AdultDataset(train_data, tokenizer)
val_dataset = AdultDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# 开始训练模型
# 开始训练模型
for batch in train_loader:
    inputs = {name: tensor.to(model.device) for name, tensor in batch.items() if name in ['input_ids', 'attention_mask']}
    labels = batch['label'].to(model.device)
    protected = batch['protected'].to(model.device)

    # 计算模型的输出
    outputs = model(**inputs)[0][:,0,:]

    # 计算原始的分类器损失
    logits = classifier(outputs)
    original_loss = cross_entropy(logits, labels)

    # 计算损失函数关于输入的梯度
    inputs['input_ids'].requires_grad = True
    original_loss.backward()

    # 应用FGSM
    sign_gradient = inputs['input_ids'].grad.data.sign()
    perturbed_input_ids = inputs['input_ids'] + epsilon * sign_gradient

    # 将对抗样本输入模型
    perturbed_outputs = model(input_ids=perturbed_input_ids, attention_mask=inputs['attention_mask'])[0][:,0,:]

    # 计算扰动后的分类器损失
    perturbed_logits = classifier(perturbed_outputs)
    perturbed_loss = cross_entropy(perturbed_logits, labels)

    # 计算MLP损失
    mlp_outputs = mlp(outputs)
    mlp_perturbed_outputs = mlp(perturbed_outputs)
    mlp_similarity_loss = 1 - cosine_similarity(mlp_outputs, mlp_perturbed_outputs).mean()

    # 计算总损失
    total_loss = original_loss + perturbed_loss + mlp_similarity_loss
    print(f"Total loss: {total_loss.item()}")

    # 反向传播和优化
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()


# text = "This is an example sentence for BERT."
# label = torch.tensor([1])  # 假设该句子的标签是1
#
# # 对输入进行编码，然后将其转换为PyTorch张量
# inputs = tokenizer(text, return_tensors='pt')
# inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
#
# # 计算模型的输出
# outputs = model(**inputs)
#
# # 对原始输出应用分类器
# classifier_outputs = classifier(outputs.logits)
#
# # 计算分类器的损失
# classifier_loss = cross_entropy(classifier_outputs, label)
#
# # 计算损失函数关于输入的梯度
# inputs['input_ids'].requires_grad = True
# classifier_loss.backward(retain_graph=True)
#
# # 应用FGSM
# epsilon = 0.01
# sign_gradient = inputs['input_ids'].grad.data.sign()
# perturbed_input_ids = inputs['input_ids'] + epsilon * sign_gradient
#
# # 将对抗样本输入模型
# perturbed_outputs = model(input_ids=perturbed_input_ids, attention_mask=inputs['attention_mask'])
#
# # 对扰动后的输出应用分类器
# perturbed_classifier_outputs = classifier(perturbed_outputs.logits)
#
# # 计算扰动后的分类器损失
# perturbed_classifier_loss = cross_entropy(perturbed_classifier_outputs, label)
#
# # 通过MLP计算相似性损失
# mlp_outputs = mlp(outputs.logits)
# mlp_perturbed_outputs = mlp(perturbed_outputs.logits)
# mlp_similarity_loss = 1 - cosine_similarity(mlp_outputs, mlp_perturbed_outputs).mean()
#
# # 计算总损失
# total_loss = classifier_loss + perturbed_classifier_loss + mlp_similarity_loss
#
# # 初始化优化器，注意这里只包括了BERT模型和分类器的参数，没有包括MLP的参数
# optimizer = Adam(list(model.parameters()) + list(classifier.parameters()))
#
# # 使用总损失进行反向传播
# total_loss.backward()
#
# # 使用优化器更新参数
# optimizer.step()
#
# print(f"Total loss: {total_loss.item()}")
