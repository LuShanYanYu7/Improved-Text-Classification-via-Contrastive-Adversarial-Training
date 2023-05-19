import torch
import subprocess
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer, BertModel, AdamW
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


# 用于计算L_ctr
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自文本，i为初始文本的embedding,j为添加扰动后的embedding
        # print("Size of mlp_outputs: ", emb_i.size())
        # print("Size of mlp_perturbed_outputs: ", emb_j.size())
        # print("Size of batch:", self.batch_size)
        z_i = nn.functional.normalize(emb_i, dim=1)
        z_j = nn.functional.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                            dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


# class ContrastiveLossELI5(nn.Module):
#     def __init__(self, batch_size, temperature=0.5, verbose=True):
#         super().__init__()
#         self.batch_size = batch_size
#         self.register_buffer("temperature", torch.tensor(temperature))
#         self.verbose = verbose
#
#     def forward(self, emb_i, emb_j):
#         """
#         emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
#         z_i, z_j as per SimCLR paper
#         """
#         print("Size of mlp_outputs: ", emb_i.size())
#         print("Size of mlp_perturbed_outputs: ", emb_j.size())
#         print("Size of batch:", self.batch_size)
#         z_i = nn.functional.normalize(emb_i, dim=1)
#         z_j = nn.functional.normalize(emb_j, dim=1)
#
#         representations = torch.cat([z_i, z_j], dim=0)
#         similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
#                                                             dim=2)
#
#         # if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
#
#         def l_ij(i, j):
#             z_i_, z_j_ = representations[i], representations[j]
#             sim_i_j = similarity_matrix[i, j]
#             # if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
#
#             numerator = torch.exp(sim_i_j / self.temperature)
#             one_for_not_i = torch.ones((2 * self.batch_size,)).to(device).scatter_(0, torch.tensor([i]), 0.0)
#             # if self.verbose: print(f"1{{k!={i}}}", one_for_not_i)
#
#             denominator = torch.sum(
#                 one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
#             )
#             # if self.verbose: print("Denominator", denominator)
#
#             loss_ij = -torch.log(numerator / denominator)
#             # if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
#
#             return loss_ij.squeeze(0)
#
#         N = self.batch_size
#         loss = 0.0
#         for k in range(0, N):
#             loss += l_ij(k, k + N) + l_ij(k + N, k)
#         return 1.0 / (2 * N) * loss


# 初始化模型和分词器
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = get_best_device()
model.to(device)

# 定义一个线性分类器
# 假设任务是二分类任务
num_labels = 2  # number of classed
classifier = nn.Linear(model.config.hidden_size, num_labels).to(device)

# 定义一个MLP
d_k = 300
mlp = nn.Sequential(
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
    nn.ReLU(),
    nn.Linear(model.config.hidden_size, d_k),
).to(device)

# 初始化优化器
# Change optimizer to AdamW with weight decay 0.01
optimizer = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=2e-5, weight_decay=0.01)

# 设置epsilon
# epsilon = [0.0001,0.001,0.005,0.02]
epsilon = 0.001


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
age_threshold = data['Age'].median()

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 请注意，BERT模型通常对自然语言文本的理解能力较强，如果特征本身足够表达它们的含义，那么在某些情况下，不包含列名可能也能取得良好的效果。
train_data['text'] = train_data.apply(combine_features, axis=1)
train_data['Age'] = (train_data['Age'] >= age_threshold).astype(int)
val_data['text'] = val_data.apply(combine_features, axis=1)
val_data['Age'] = (val_data['Age'] >= age_threshold).astype(int)

train_dataset = AdultDataset(train_data, tokenizer)
val_dataset = AdultDataset(val_data, tokenizer)

# batch_size = 8
batch_size = 16
# batch_size = 32
# drop_last 丢弃多余的数据，以免最后一个batch数据大小对不上
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

epochs = 3
# 开始训练模型
accuracy_list = []
DAO_list = []
DEO_list = []
loss_list = []

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
        inputs = {name: tensor.to(model.device) for name, tensor in batch.items() if
                  name in ['input_ids', 'attention_mask']}
        labels = batch['label'].to(model.device)
        protected = batch['protected'].to(model.device)

        # 计算模型的输出
        outputs = model(**inputs)[0][:, 0, :]

        # 计算原始的分类器损失
        logits = classifier(outputs)
        original_loss = nn.functional.cross_entropy(logits, labels)

        # 计算损失函数关于输入的梯度
        model.embeddings.word_embeddings.weight.requires_grad = True
        original_loss.backward(retain_graph=True)

        # 应用FGSM
        normalized_gradient = torch.nn.functional.normalize(model.embeddings.word_embeddings.weight.grad.data, p=2)
        perturbed_embeddings = model.embeddings.word_embeddings.weight + epsilon * normalized_gradient

        # 将对抗样本输入模型
        model.embeddings.word_embeddings.weight.data = perturbed_embeddings
        perturbed_outputs = model(**inputs)[0][:, 0, :]

        # 计算扰动后的分类器损失
        perturbed_logits = classifier(perturbed_outputs)
        perturbed_loss = nn.functional.cross_entropy(perturbed_logits, labels)

        # 计算MLP损失
        mlp_outputs = mlp(outputs)
        mlp_perturbed_outputs = mlp(perturbed_outputs)
        # 温度系数
        # t=[0.05,0.06,0.07,0.08,0.09,0.10]
        t = 0.05
        loss_func = ContrastiveLoss(batch_size=batch_size, temperature=t)
        mlp_similarity_loss = loss_func(mlp_outputs, mlp_perturbed_outputs)

        # loss_eli5 = ContrastiveLossELI5(batch_size=batch_size, temperature=t, verbose=True)
        # mlp_similarity_loss = loss_eli5(mlp_outputs, mlp_perturbed_outputs)

        # 计算总损失
        # Lambda = [0.1,0.2,0.3,0.4,0.5]
        Lambda = 0.5
        total_batch_loss = (1 - Lambda) * (original_loss + perturbed_loss) / 2 + mlp_similarity_loss * Lambda
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
        y_real = {'income': labels, 'Age': protected}  # 假设'Age'是敏感特征
        privileged = 1
        unprivileged = 0

        batch_DEO = DifferenceEqualOpportunity(y_pred, y_real, 'Age', 'income', privileged, unprivileged, [0, 1])
        batch_DAO = DifferenceAverageOdds(y_pred, y_real, 'Age', 'income', privileged, unprivileged, [0, 1])

        total_DAO += batch_DAO
        total_DEO += batch_DEO

        # 打印每个批次的信息
        print(
            f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {total_batch_loss.item()}, Accuracy: {correct / labels.size(0)}, DAO: {batch_DAO}, DEO: {batch_DEO}")

    # 每个epoch结束后，保存模型
    torch.save(model.state_dict(), f"result/model_epoch_{epoch}.pt")

    # 计算并打印平均损失
    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss}")

    # 计算并打印准确率
    avg_accuracy = total_correct / total_examples
    accuracy_list.append(avg_accuracy)
    print(f"Epoch {epoch + 1}/{epochs}, Avg Accuracy: {avg_accuracy}")

    # 计算并打印DAO和DEO
    avg_DAO = total_DAO / len(train_loader)
    avg_DEO = total_DEO / len(train_loader)
    DAO_list.append(avg_DAO)
    DEO_list.append(avg_DEO)
    print(f"Epoch {epoch + 1}/{epochs}, Avg DAO: {avg_DAO}, Avg DEO: {avg_DEO}")

# 创建一个名为 "result" 的文件夹，如果它不存在的话
if not os.path.exists('result'):
    os.makedirs('result')

# loss
plt.figure()
plt.plot(range(epochs), accuracy_list)
plt.title('loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.savefig('result/loss.png')  # Save the figure
plt.close()

# Accuracy
plt.figure()
plt.plot(range(epochs), accuracy_list)
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('result/accuracy.png')  # Save the figure
plt.close()

# DAO
plt.figure()
plt.plot(range(epochs), DAO_list)
plt.title('DAO over epochs')
plt.xlabel('Epochs')
plt.ylabel('DAO')
plt.savefig('result/DAO.png')  # Save the figure
plt.close()

# DEO
plt.figure()
plt.plot(range(epochs), DEO_list)
plt.title('DEO over epochs')
plt.xlabel('Epochs')
plt.ylabel('DEO')
plt.savefig('result/DEO.png')  # Save the figure
plt.close()
