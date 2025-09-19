# ufld_infer_crop_fixed.py
import os
from PIL import Image
import numpy as np
import cv2
import torch
import scipy.special, tqdm
import torchvision.transforms as transforms

# ---------------------------
# 固定設定
# ---------------------------
FRAMES_DIR = "dataset/run_1756133797/frames"
MODEL_PATH = "model/tusimple_18.pth"
OUT_VIDEO = "ufld_output.avi"
OUT_FRAMES = "out"   # 設成 "" 代表不要存逐張輸出
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "18"
GRIDING_NUM = 100
CLS_NUM_PER_LANE = 56   # Tusimple
USE_AUX = False

# 裁切比例
TOP_RATIO = 0.2
BOTTOM_RATIO = 0.4
LEFT_RATIO = 0.35
RIGHT_RATIO = 0.3
# ---------------------------

# 讀取 frames
frames = sorted([os.path.join(FRAMES_DIR, f) for f in os.listdir(FRAMES_DIR)
                 if f.lower().endswith(".jpg") or f.lower().endswith(".png")])
if len(frames) == 0:
    raise RuntimeError("No image frames found in %s" % FRAMES_DIR)
print(f"Found {len(frames)} frames.")

# 讀第一張圖片決定大小
first_frame = cv2.imread(frames[0])
if first_frame is None:
    raise RuntimeError("Failed to read first frame: %s" % frames[0])
IMG_H, IMG_W = first_frame.shape[:2]
print(f"Image size detected: {IMG_W}x{IMG_H}")

# ---------------------------
# 載入官方 model
# ---------------------------
from model.model import parsingNet

cls_dim = (GRIDING_NUM+1, CLS_NUM_PER_LANE, 4)
net = parsingNet(size=(288,800), pretrained=False, backbone=BACKBONE, cls_dim=cls_dim, use_aux=USE_AUX)
net.to(DEVICE)

# 載入權重
print("Loading model:", MODEL_PATH)
ckpt = torch.load(MODEL_PATH, map_location="cpu")
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
net.load_state_dict(state_dict, strict=False)
net.eval()

# ---------------------------
# transforms
# ---------------------------
img_transforms = transforms.Compose([
    transforms.Resize((288,800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229,0.224,0.225)),
])

# ---------------------------
# row anchor
# ---------------------------
row_anchor = np.linspace(0, 287, CLS_NUM_PER_LANE)

# ---------------------------
# VideoWriter
# ---------------------------
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vout = cv2.VideoWriter(OUT_VIDEO, fourcc, 30.0, (IMG_W, IMG_H))
if OUT_FRAMES:
    os.makedirs(OUT_FRAMES, exist_ok=True)

col_sample = np.linspace(0, 800-1, GRIDING_NUM)
col_sample_w = col_sample[1]-col_sample[0]

# ---------------------------
# 推論 loop
# ---------------------------
for idx_img, img_path in enumerate(tqdm.tqdm(frames)):
    pil_img = Image.open(img_path).convert("RGB")
    
    # ---------------------------
    # 自訂裁切
    # ---------------------------
    w, h = pil_img.size
    left = int(w * LEFT_RATIO)
    right = int(w * (1 - RIGHT_RATIO))
    top = int(h * TOP_RATIO)
    bottom = int(h * (1 - BOTTOM_RATIO))
    pil_img_crop = pil_img.crop((left, top, right, bottom))  # (left, top, right, bottom)

    img_t = img_transforms(pil_img_crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = net(img_t)

    out_j = out[0].cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1,:,:], axis=0)
    idx = np.arange(GRIDING_NUM)+1
    idx = idx.reshape(-1,1,1)
    loc = np.sum(prob*idx, axis=0)
    out_max = np.argmax(out_j, axis=0)
    loc[out_max==GRIDING_NUM] = 0

    # 原圖用於顯示
    vis = cv2.imread(img_path)
    if vis is None:
        vis = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ---------------------------
    # 映射 lane 座標回原圖
    # ---------------------------
    crop_h = bottom - top
    crop_w = right - left
    scale_y = crop_h / 288.0
    scale_x = crop_w / 800.0

    for lane_idx in range(out_max.shape[1]):
        if np.sum(out_max[:,lane_idx]!=0) > 2:
            for k in range(out_max.shape[0]):
                if out_max[k,lane_idx]>0:
                    x = loc[k,lane_idx]*col_sample_w*scale_x + left
                    y = row_anchor[CLS_NUM_PER_LANE-1-k]*scale_y + top
                    x = int(max(0,min(IMG_W-1,x)))
                    y = int(max(0,min(IMG_H-1,y)))
                    cv2.circle(vis,(x,y),5,(0,255,0),-1)

    vout.write(vis)
    if OUT_FRAMES:
        out_path = os.path.join(OUT_FRAMES,f"{idx_img:06d}.jpg")
        cv2.imwrite(out_path, vis)

vout.release()
print("Done. Video saved to:", OUT_VIDEO)
if OUT_FRAMES:
    print("Frames saved to:", OUT_FRAMES)
