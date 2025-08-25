import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ETS2Dataset
from model import LaneNet

def train_model():
    # 排除測試集 run_1756133797
    dataset = ETS2Dataset("dataset", target_size=(256,128), exclude_runs=["run_1756133797"])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LaneNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "lane_model2.pth")
    print("✅ 模型已存成 lane_model2.pth")

if __name__ == "__main__":
    train_model()
