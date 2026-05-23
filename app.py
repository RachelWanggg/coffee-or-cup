import gradio as gr

import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image

# device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# classes
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

# load model
model.load_state_dict(
    torch.load("model.pth")
)

model = model.to(device)

model.eval()

# prediction function
def predict(image):

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

# gradio UI
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Coffee vs Cup Classifier"
)

app.launch()