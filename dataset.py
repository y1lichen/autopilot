import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def letterbox_resize(img, target_size):
    """
    等比例縮小，並以黑邊 padding 到固定尺寸 (W, H)
    target_size: (target_w, target_h)
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]

    # 等比例縮小到最長邊貼近 target
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 建立黑底畫布，將縮好後的圖貼上（置中）
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


class ETS2Dataset(Dataset):
    def __init__(self, frames_dir, csv_file, target_size=(320, 160), keep_top_ratio=0.65, transform=None):
        """
        frames_dir: frames 資料夾
        csv_file  : actions.csv（至少要有 frame_id, steer）
                    steer 應為 -1(左/A), 0(直), 1(右/D)
        target_size: (W, H) 模型固定輸入尺寸；內部做等比例縮放 + padding
        keep_top_ratio: 先裁掉下半部，只保留上方比例 (e.g. 0.65 表示保留上面 65%)
        transform: 額外的 tensor 層級 transform（通常可留 None）
        """
        self.frames_dir = frames_dir
        self.df = pd.read_csv(csv_file)
        self.target_size = target_size
        self.keep_top_ratio = float(keep_top_ratio)
        self.transform = transform

        # 只保留實體存在的檔案
        self.df["filepath"] = self.df["frame_id"].apply(lambda x: os.path.join(frames_dir, x))
        self.df = self.df[self.df["filepath"].apply(os.path.exists)].reset_index(drop=True)

        # 檢查必要欄位
        for col in ["frame_id", "steer"]:
            if col not in self.df.columns:
                raise ValueError(f"CSV 少欄位: {col}")

        # 將 steer 值轉成 class index（CrossEntropy 要從 0 開始）
        # -1 -> 0 (left), 0 -> 1 (straight), 1 -> 2 (right)
        self.df["label"] = self.df["steer"].map({-1: 0, 0: 1, 1: 2})
        if self.df["label"].isna().any():
            raise ValueError("CSV 的 steer 欄位必須只包含 -1, 0, 1")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]

        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise FileNotFoundError(f"{img_path} not found")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 裁切：只保留上半部（或上方 keep_top_ratio）
        h, w, _ = img.shape
        keep_h = int(h * self.keep_top_ratio)
        img = img[:keep_h, :, :]

        # 等比例縮到 target_size，並 padding
        img = letterbox_resize(img, self.target_size)  # (H, W, 3) uint8

        # 正規化到 [0,1] 並轉 tensor (C,H,W)
        img = img.astype("float32") / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)

        # 額外 transform（如果有）
        if self.transform:
            img_tensor = self.transform(img_tensor)

        label = int(row["label"])  # 0/1/2
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor

if __name__ == "__main__":
    dataset = ETS2Dataset(
        frames_dir="dataset/run_1755875273/frames",
        csv_file="dataset/run_1755875273/actions.csv",
    )

    n_show = 5
    indices = random.sample(range(len(dataset)), n_show)

    for idx in indices:
        img_tensor, label = dataset[idx]  # Tensor: (C,H,W)

        # ---- 轉回可顯示圖片 ----
        img = img_tensor.permute(1, 2, 0).numpy()  # (H,W,C), float
        # 假設 tensor 值在 [0,1]，轉回 [0,255]
        img = (img * 255).clip(0, 255).astype("uint8")
        # RGB -> BGR (OpenCV顯示需要)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ---- 顯示 ----
        cv2.imshow(f"label: {label}", img)
        cv2.waitKey(0)  # 按任意鍵看下一張

    cv2.destroyAllWindows()
