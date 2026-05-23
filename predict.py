import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image

# device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# 类别
classes = ['coffee', 'cup']

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# model
model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)

# 加载模型
model.load_state_dict(
    torch.load("model.pth")
)

model = model.to(device)

# 推理模式
model.eval()

# 读取测试图片
image = Image.open("test.jpeg")

image = transform(image)

# 增加 batch 维度
image = image.unsqueeze(0)

image = image.to(device)

# inference
with torch.no_grad():

    outputs = model(image)

    _, predicted = torch.max(outputs, 1)

print("Prediction:", classes[predicted.item()])