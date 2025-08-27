# test.py
import torch
from torch.utils.data import DataLoader
from dataset import ETS2Dataset
from model import LaneNet
import cv2
import os
from collections import defaultdict

def test_model(model_path="lane_model2.pth", dataset_dir="dataset", test_run="run_1756133797"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 載入模型
    model = LaneNet(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 只載入 test_run
    dataset = ETS2Dataset(
        dataset_dir=dataset_dir,
        target_size=(256,128),
        keep_top_ratio=0.65,
        exclude_runs=[r for r in os.listdir(dataset_dir) if r != test_run]
    )
    dataset.files = [f for f in dataset.files if test_run in f[0]]
    print(f"測試集總共有 {len(dataset)} 張圖片")

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    # 每類的正確數 / 總數
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    show_n = 1
    shown = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for l, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                per_class_total[l] += 1
                if l == p:
                    per_class_correct[l] += 1

            # 顯示前幾張圖片
            if shown < show_n:
                for i in range(min(show_n - shown, imgs.size(0))):
                    img = imgs[i].cpu().permute(1,2,0).numpy()
                    img = (img * 255).astype("uint8")
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    pred_label = preds[i].item()
                    true_label = labels[i].item()
                    cv2.imshow(f"pred:{pred_label}, true:{true_label}", img)
                    cv2.waitKey(0)
                    shown += 1

    # 整體準確度
    acc = correct / total if total > 0 else 0
    print(f"\nOverall Test Accuracy: {acc:.4f}")

    # 各類別準確率
    label_map = {0: -1, 1: 0, 2: 1}  # 對應回原始標籤
    for c in sorted(per_class_total.keys()):
        acc_c = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0
        print(f"Class {label_map[c]} Accuracy: {acc_c:.4f} ({per_class_correct[c]}/{per_class_total[c]})")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model()
