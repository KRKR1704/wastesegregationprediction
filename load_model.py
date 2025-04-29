import torch
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild model structure
num_classes = 10
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model/image_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

print(" Model loaded and ready for inference!")
