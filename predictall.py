import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights
)
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#dataset
dataset_path = "dataset"
dummy_dataset = ImageFolder(dataset_path)
class_names = dummy_dataset.classes

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#ResNet50
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(class_names))
resnet.load_state_dict(torch.load("model/image_classifier.pth", map_location=device))
resnet = resnet.to(device)
resnet.eval()

#EfficientNet-B0
efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
efficientnet.classifier[1] = torch.nn.Linear(efficientnet.classifier[1].in_features, len(class_names))
efficientnet.load_state_dict(torch.load("model/efficientnetb0_classifier.pth", map_location=device))
efficientnet = efficientnet.to(device)
efficientnet.eval()

#MobileNetV3
mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
mobilenet.classifier[3] = torch.nn.Linear(mobilenet.classifier[3].in_features, len(class_names))
mobilenet.load_state_dict(torch.load("model/mobilenetv3_classifier.pth", map_location=device))
mobilenet = mobilenet.to(device)
mobilenet.eval()

# Predict function
def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]


img_path = "example2.jpg"
if os.path.exists(img_path):
    print(f"Predicting for: {img_path}")
    print(f"ResNet50 Prediction:{predict_image(resnet, img_path)}")
    print(f"EfficientNet-B0 Prediction:{predict_image(efficientnet, img_path)}")
    print(f"MobileNetV3 Prediction:{predict_image(mobilenet, img_path)}")
else:
    print(f" File not found: {img_path}")
