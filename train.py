import torch, json
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    BASE = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(f"{BASE}/train", transform=train_tf)
    val_set   = datasets.ImageFolder(f"{BASE}/valid", transform=val_tf)
    train_dl = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_set, batch_size=32, num_workers=2)

    json.dump(train_set.classes, open("class_names.json", "w"))
    print(len(train_set.classes), "classes")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_set.classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    def evaluate():
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
        return correct / total

    def train_epochs(n, optimizer, phase):
        for epoch in range(n):
            model.train()
            for i, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward(); optimizer.step()
                if i % 200 == 0:
                    print(f"  [{phase}] epoch {epoch+1} batch {i}/{len(train_dl)} loss {loss.item():.3f}")
            print(f"[{phase}] Epoch {epoch+1}: val acc = {evaluate():.4f}")

    # Phase 1: freeze backbone, train head only
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    train_epochs(2, optim.Adam(model.fc.parameters(), lr=1e-3), "head")

    # Phase 2: unfreeze everything, fine-tune with a low LR
    for p in model.parameters():
        p.requires_grad = True
    train_epochs(3, optim.Adam(model.parameters(), lr=1e-4), "fine-tune")

    torch.save(model.state_dict(), "model.pth")
    print("Saved model.pth, final val acc =", round(evaluate(), 4))

if __name__ == "__main__":
    main()
