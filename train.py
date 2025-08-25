# train.py
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2

# ---------------- Dataset ---------------- #
class ETS2Dataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: e.g. "dataset"
        會自動讀取所有 run_xxxx 資料夾
        """
        self.samples = []

        run_folders = glob.glob(os.path.join(root_dir, "run_*"))
        print(f"找到 {len(run_folders)} 個 run_* 資料夾")

        for run_dir in run_folders:
            frames_dir = os.path.join(run_dir, "frames")
            csv_file = os.path.join(run_dir, "actions.csv")
            if not os.path.exists(csv_file):
                print(f"⚠️ 跳過 {run_dir}，沒有 actions.csv")
                continue

            df = pd.read_csv(csv_file)
            if "frame_id" not in df.columns or "steer" not in df.columns:
                print(f"⚠️ {csv_file} 缺少必要欄位，跳過")
                continue

            for _, row in df.iterrows():
                img_path = os.path.join(frames_dir, row["frame_id"])
                if os.path.exists(img_path):  # 避免缺檔
                    # label 直接轉 Tensor
                    label = torch.tensor(int(row["steer"]), dtype=torch.long)
                    self.samples.append((img_path, label))

        print(f"✅ 總共載入 {len(self.samples)} 筆樣本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 裁掉下半部，保留上 65%
        h, w, _ = img.shape
        img = img[:int(h * 0.65), :, :]

        # resize 成固定大小
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, label

# ---------------- Model ---------------- #
class LaneNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))  # 強制 CNN 輸出 (128,16,16)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

# ---------------- Train ---------------- #
def train_model():
    dataset = ETS2Dataset("dataset")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = LaneNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
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
    print("✅ 模型已存成 lane_model.pth")

if __name__ == "__main__":
    train_model()
