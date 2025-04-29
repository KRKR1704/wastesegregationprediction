import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np
from tqdm import tqdm

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    #Loading dataset
    dataset_root = 'dataset'
    full_dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    targets = [label for _, label in full_dataset.imgs]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    #MobileNetV3
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    #Training loop
    epochs = 25
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"\nTraining Epoch {epoch+1}/{epochs}")
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        scheduler.step(average_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")

        if average_loss < best_loss:
            best_loss = average_loss
            patience_counter = 0
            os.makedirs("model", exist_ok=True)
            torch.save(model.state_dict(), "model/mobilenetv3_classifier.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(" Training complete. Best model saved.")

    model.eval()
    all_preds = []
    all_labels = []

    print("\n Evaluating model on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    #Accuracy
    accuracy = 100 * np.mean(all_preds == all_labels)
    print(f"\n Final Test Accuracy: {accuracy:.2f}%")

    print("\n Per-Class Precision, Recall, F1:")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    for cls in sorted(set(class_names)):
        if cls in report:
            precision = report[cls]['precision']
            recall = report[cls]['recall']
            f1 = report[cls]['f1-score']
            print(f"{cls}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n Overall (Macro Avg):")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall:    {macro_recall:.4f}")
    print(f"F1 Score:  {macro_f1:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
