import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import random


class ETS2Dataset(Dataset):
    def __init__(self, frames_dir, csv_file, img_size=None, transform=None):
        """
        frames_dir : str, 存放截圖的資料夾
        csv_file : str, actions.csv 路徑
        img_size : tuple, 縮小後尺寸 (width, height)
        transform : callable, 可選，資料增強
        """
        self.frames_dir = frames_dir
        self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.frames_dir, row['frame_id'])
        
        # 讀取圖片
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise FileNotFoundError(f"{img_path} not found")
        
        # 轉 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 裁切下半部
        h, w, _ = img.shape
        img = img[:int(h*0.75):, :, :]
        
        # 縮小到指定尺寸
        if self.img_size:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)  # 保留細節
        
        # 正規化到 [0,1]
        img = img.astype('float32') / 255.0
        
        # 轉成 Tensor, (C,H,W)
        img_tensor = torch.tensor(img).permute(2,0,1)
        
        # label
        label = torch.tensor(row['steer'], dtype=torch.long)
        
        # 資料增強 (可選)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label

# 使用範例
if __name__ == "__main__":
    dataset = ETS2Dataset(frames_dir="dataset/run_1755875273/frames",
                          csv_file="dataset/run_1755875273/actions.csv",
                          )

    n_show = 5
    indices = random.sample(range(len(dataset)), n_show)

    for idx in indices:
        img_tensor, label = dataset[idx]  # Tensor (C,H,W)
        # 轉回 OpenCV 可顯示格式 (H,W,C) 並轉成 uint8
        img = img_tensor.permute(1,2,0).numpy() * 255
        img = img.astype('uint8')
        
        # 將 label 轉成文字
        
        # 顯示圖片
        cv2.imshow(f"{label}", img)
        cv2.waitKey(0)  # 按任意鍵顯示下一張

    cv2.destroyAllWindows()
    
    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=16, shuffle=True)

