import torch, json
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    classes = json.load(open("class_names.json"))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_set = datasets.ImageFolder(f"{BASE}/valid", transform=tf)
    val_dl = DataLoader(val_set, batch_size=64, num_workers=2)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Validation accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    plt.imshow(cm, cmap="Greens")
    plt.title(f"Confusion Matrix (acc {acc:.3f})")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90, fontsize=6)
    plt.yticks(ticks, classes, fontsize=6)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    main()
