import os
import cv2
import numpy as np
from ultralytics import YOLO

# === 載入 YOLO segmentation 模型 ===
model = YOLO("yolo_weight/best.onnx", task="segment")

# === 影片幀資料夾 ===
frames_dir = "dataset/run_1755702281/frames"
frames_dir = "dataset/run_1755702912/frames" # night
frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])
if not frame_files:
    raise RuntimeError("No frames found!")

first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]

# === 影片輸出設定 ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_lane_yolo_onnx.mp4", fourcc, 20.0, (width, height))

# === 推論區域設定（只保留上方 65% 高度） ===
crop_y = int(height * 0.65)

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue

    # 裁切上方 65% 畫面
    crop_frame = frame[:crop_y, :, :]

    # YOLO segmentation 推論
    results = model.predict(source=crop_frame, save=False, show=False, verbose=False)
    result = results[0]

    # 如果有多條 lane mask，只保留最靠近畫面中心的
    masks = result.masks.data  # shape: (N, H, W)
    if masks is not None and len(masks) > 0:
        img_center_x = crop_frame.shape[1] // 2

        # 計算每條 mask 的水平中心
        cx_list = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                cx_list.append(np.inf)
            else:
                cx_list.append(np.mean(xs))

        # 找最靠近中心的 mask
        best_idx = np.argmin(np.abs(np.array(cx_list) - img_center_x))
        best_mask = masks[best_idx]

        # 只保留這條 mask
        masks = np.expand_dims(best_mask, axis=0)
        result.masks.data = masks

    # 繪製 mask 與 bounding box
    crop_with_mask = result.plot()

    # 疊回完整影像
    img_with_mask = frame.copy()
    img_with_mask[:crop_y, :, :] = crop_with_mask

    # 寫入影片
    out.write(img_with_mask)

    # 顯示影像
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection", 960, 540)
    cv2.imshow("Lane Detection", img_with_mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cv2.destroyAllWindows()
