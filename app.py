from flask import Flask, render_template, request
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V3_Large_Weights, ResNet50_Weights

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load class names from dataset structure
from torchvision.datasets import ImageFolder
dummy = ImageFolder('dataset')  # dataset folder used only to get class labels
class_names = dummy.classes

# Transform: input size 224x224 for all models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
def load_model(model_name, path):
    if model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(class_names))
    
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Load models from the "model" directory
resnet = load_model("resnet50", "model/image_classifier.pth")
efficientnet = load_model("efficientnetb0", "model/efficientnetb0_classifier.pth")
mobilenet = load_model("mobilenetv3", "model/mobilenetv3_classifier.pth")

# Prediction logic
def predict(model, img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['image']
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # Predictions from all three models
            pred_resnet = predict(resnet, img_path)
            pred_efficientnet = predict(efficientnet, img_path)
            pred_mobilenet = predict(mobilenet, img_path)

            return render_template("index.html", img_path=img_path,
                                   resnet=pred_resnet,
                                   efficientnet=pred_efficientnet,
                                   mobilenet=pred_mobilenet)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
