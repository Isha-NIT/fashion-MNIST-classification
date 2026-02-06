from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import FashionMLP

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Fashion-MNIST API is running "}


# Load model
model = FashionMLP()
model.load_state_dict(torch.load("fashion_mnist.pth", map_location="cpu"))
model.eval()  # (turns off dropout)

classes = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),        # If input image might be RGB
    transforms.Resize((28, 28)),   # Must match Fashion-MNIST
    transforms.ToTensor(),         # 0–255 → 0–1
    transforms.Normalize((0.2860,), (0.3530,))
])


# Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image = transform(image).unsqueeze(0)  # shape: (1, 1, 28, 28)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }

