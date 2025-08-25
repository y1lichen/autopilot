import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def letterbox_resize(img, target_size):
    """等比例縮放 + padding 到固定尺寸 (W,H)"""
    target_w, target_h = target_size
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas

class ETS2Dataset(Dataset):
    def __init__(self, dataset_dir, target_size=(256,128), keep_top_ratio=0.65, transform=None, exclude_runs=None):
        """
        dataset_dir: dataset 主資料夾
        target_size: (W,H)
        keep_top_ratio: 裁掉下方比例
        transform: 額外 Tensor transform
        exclude_runs: list, 不包含的 run_* 資料夾
        """
        self.target_size = target_size
        self.keep_top_ratio = keep_top_ratio
        self.transform = transform
        self.files = []

        exclude_runs = exclude_runs or []

        # 找所有 run_* 資料夾
        run_dirs = [d for d in os.listdir(dataset_dir)
                    if d.startswith("run_") and os.path.isdir(os.path.join(dataset_dir,d))
                    and d not in exclude_runs]
        print(f"找到 {len(run_dirs)} 個 run_* 資料夾")

        for run in run_dirs:
            frames_dir = os.path.join(dataset_dir, run, "frames")
            csv_file = os.path.join(dataset_dir, run, "actions.csv")
            if not os.path.exists(csv_file) or not os.path.exists(frames_dir):
                continue

            df = pd.read_csv(csv_file)
            if "frame_id" not in df.columns or "steer" not in df.columns:
                continue

            df["filepath"] = df["frame_id"].apply(lambda x: os.path.join(frames_dir, x))
            df = df[df["filepath"].apply(os.path.exists)]
            df["label"] = df["steer"].map({-1:0, 0:1, 1:2})
            df = df.dropna(subset=["label"])
            self.files.extend(list(df[["filepath","label"]].itertuples(index=False, name=None)))

        print(f"總樣本數: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"{img_path} not found")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 裁上方
        h = img.shape[0]
        keep_h = int(h * self.keep_top_ratio)
        img = img[:keep_h, :, :]

        img = letterbox_resize(img, self.target_size)
        img = img.astype("float32") / 255.0
        img_tensor = torch.from_numpy(img).permute(2,0,1)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor
