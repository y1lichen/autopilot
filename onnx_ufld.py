import os, cv2, numpy as np, tqdm
from PIL import Image
import onnxruntime as ort
import scipy.special

# ---------------------------
FRAMES_DIR = "dataset/run_1756133797/frames"
ONNX_MODEL = "model/ufld_tusimple_18.onnx"
OUT_VIDEO = "ufld_output_onnx.avi"
OUT_FRAMES = "out_frames"  # "" = 不存逐張
GRIDING_NUM = 100
CLS_NUM_PER_LANE = 56
# 裁切比例
TOP_RATIO = 0.3
BOTTOM_RATIO = 0.4
LEFT_RATIO = 0.0
RIGHT_RATIO = 0.0
# ---------------------------

# 讀 frames
frames = sorted([os.path.join(FRAMES_DIR,f) for f in os.listdir(FRAMES_DIR)
                 if f.lower().endswith(".jpg") or f.lower().endswith(".png")])
if len(frames)==0: raise RuntimeError("No frames found")

# 讀第一張圖
first_frame = cv2.imread(frames[0])
IMG_H, IMG_W = first_frame.shape[:2]

# ---------------------------
# ONNX session with CPU parallel BLAS
# ---------------------------
providers = ["CPUExecutionProvider"]
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = os.cpu_count()  # 設成 CPU 核心數
session = ort.InferenceSession(ONNX_MODEL, sess_options=sess_options, providers=providers)

# transforms
def preprocess(pil_img):
    img = pil_img.resize((800,288))
    img = np.array(img).astype(np.float32)/255.0
    img = (img - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    img = img.transpose(2,0,1)  # HWC -> CHW
    return np.expand_dims(img,0).astype(np.float32)

# VideoWriter (原圖大小)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vout = cv2.VideoWriter(OUT_VIDEO, fourcc, 30.0, (IMG_W, IMG_H))
if OUT_FRAMES: os.makedirs(OUT_FRAMES,exist_ok=True)

col_sample = np.linspace(0, 800-1, GRIDING_NUM)
col_sample_w = col_sample[1]-col_sample[0]
row_anchor = np.linspace(0, 287, CLS_NUM_PER_LANE)

for idx_img, img_path in enumerate(tqdm.tqdm(frames)):
    pil_img = Image.open(img_path).convert("RGB")
    w,h = pil_img.size
    left = int(w*LEFT_RATIO)
    right = int(w*(1-RIGHT_RATIO))
    top = int(h*TOP_RATIO)
    bottom = int(h*(1-BOTTOM_RATIO))

    # 裁切後影像
    pil_crop = pil_img.crop((left,top,right,bottom))
    img_t = preprocess(pil_crop)

    # ONNX 推理
    out = session.run(None, {"input": img_t})[0]
    out_j = out[0]  # shape: (grid+1, cls_per_lane, lanes)
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1,:,:], axis=0)
    idx = np.arange(GRIDING_NUM)+1
    idx = idx.reshape(-1,1,1)
    loc = np.sum(prob*idx, axis=0)
    out_max = np.argmax(out_j, axis=0)
    loc[out_max==GRIDING_NUM]=0

    # ====== 原圖繪製 ======
    vis_full = cv2.imread(img_path)
    if vis_full is None:
        vis_full = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ====== 裁切圖繪製 ======
    vis_crop = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)

    crop_h = bottom-top
    crop_w = right-left
    scale_y = crop_h/288.0
    scale_x = crop_w/800.0

    for lane_idx in range(out_max.shape[1]):
        if np.sum(out_max[:,lane_idx]!=0)>2:
            for k in range(out_max.shape[0]):
                if out_max[k,lane_idx]>0:
                    # 對應到裁切後座標
                    x_crop = loc[k,lane_idx]*col_sample_w*scale_x
                    y_crop = row_anchor[CLS_NUM_PER_LANE-1-k]*scale_y
                    x_crop = int(max(0,min(crop_w-1,x_crop)))
                    y_crop = int(max(0,min(crop_h-1,y_crop)))
                    cv2.circle(vis_crop,(x_crop,y_crop),5,(0,255,0),-1)

                    # 對應到原圖座標
                    x_full = x_crop + left
                    y_full = y_crop + top
                    cv2.circle(vis_full,(x_full,y_full),5,(0,255,0),-1)

    # 存 AVI (完整原圖)
    vout.write(vis_full)

    # 存逐張 frame (裁切後圖)
    if OUT_FRAMES:
        out_path = os.path.join(OUT_FRAMES,f"{idx_img:06d}.jpg")
        cv2.imwrite(out_path, vis_crop)

vout.release()
print("Done. Video saved to:", OUT_VIDEO)
if OUT_FRAMES: print("Frames saved to:", OUT_FRAMES)
