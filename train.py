import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 检测 Metal GPU
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 读取数据集
dataset = datasets.ImageFolder(
    "dataset",
    transform=transform
)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# 加载预训练 ResNet18
model = models.resnet18(pretrained=True)

# 修改最后一层
num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)

# 放到 GPU
model = model.to(device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# 开始训练
epochs = 3

for epoch in range(epochs):

    running_loss = 0.0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# 保存模型
torch.save(
    model.state_dict(),
    "model.pth"
)

print("Model saved!")