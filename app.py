import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0)  

# Load your model
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(128 * 16 * 16, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 4)  # 4 classes
)

model.load_state_dict(torch.load("model1.pt", map_location="cpu"))
model.eval()

# Define your class labels
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Prediction function
def predict(image):
    image = image.convert("RGB")
    tensor = transform_image(image)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(1).item()
        return f"Predicted Class: {class_names[pred]}"

# Launch Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs="text",
    title="Image Classifier",
    description="Upload an image to classify it."
)

if __name__ == "__main__":
    interface.launch()
