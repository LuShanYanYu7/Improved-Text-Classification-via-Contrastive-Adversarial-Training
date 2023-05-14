import torch
from torch.nn.functional import cross_entropy, cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
from torch.optim import Adam

# 初始化模型和分词器
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 定义一个线性分类器
classifier = nn.Linear(model.config.hidden_size, 2)  # 假设任务是二分类任务

# 定义一个MLP
mlp = nn.Sequential(
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
    nn.ReLU(),
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
)

# 定义输入文本和标签
text = "This is an example sentence for BERT."
label = torch.tensor([1])  # 假设该句子的标签是1

# 对输入进行编码，然后将其转换为PyTorch张量
inputs = tokenizer(text, return_tensors='pt')
inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

# 计算模型的输出
outputs = model(**inputs)

# 对原始输出应用分类器
classifier_outputs = classifier(outputs.logits)

# 计算分类器的损失
classifier_loss = cross_entropy(classifier_outputs, label)

# 计算损失函数关于输入的梯度
inputs['input_ids'].requires_grad = True
classifier_loss.backward(retain_graph=True)

# 应用FGSM
epsilon = 0.01
sign_gradient = inputs['input_ids'].grad.data.sign()
perturbed_input_ids = inputs['input_ids'] + epsilon * sign_gradient

# 将对抗样本输入模型
perturbed_outputs = model(input_ids=perturbed_input_ids, attention_mask=inputs['attention_mask'])

# 对扰动后的输出应用分类器
perturbed_classifier_outputs = classifier(perturbed_outputs.logits)

# 计算扰动后的分类器损失
perturbed_classifier_loss = cross_entropy(perturbed_classifier_outputs, label)

# 通过MLP计算相似性损失
mlp_outputs = mlp(outputs.logits)
mlp_perturbed_outputs = mlp(perturbed_outputs.logits)
mlp_similarity_loss = 1 - cosine_similarity(mlp_outputs, mlp_perturbed_outputs).mean()

# 计算总损失
total_loss = classifier_loss + perturbed_classifier_loss + mlp_similarity_loss

# 初始化优化器，注意这里只包括了BERT模型和分类器的参数，没有包括MLP的参数
optimizer = Adam(list(model.parameters()) + list(classifier.parameters()))

# 使用总损失进行反向传播
total_loss.backward()

# 使用优化器更新参数
optimizer.step()

print(f"Total loss: {total_loss.item()}")
