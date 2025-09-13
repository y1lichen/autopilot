import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# === 載入 YOLO segmentation 模型 ===
# Load the YOLO segmentation model
model = YOLO("yolo_weight/best.onnx", task="segment")

# === 影片幀資料夾 ===
# Video frame folder
frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])
if not frame_files:
    raise RuntimeError("找不到任何影片幀！")

first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]

# === 影片輸出設定 ===
# Video output settings
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_lane_yolo_onnx.mp4", fourcc, 20.0, (width, height))

# === 推論區域設定（只保留上方 65% 高度） ===
# Inference area setting (only top 65% height)
crop_y = int(height * 0.65)

triangle_offset = 50 # 三角形水平偏移參數（像素，正數往右，負數往左）
# 根據三角形偏移量設定車輛中心位置
# Set the vehicle center position based on the triangle offset
vehicle_center_x = width // 2 + triangle_offset

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue

    # 裁切上方 65% 畫面
    # Crop the top 65% of the frame
    crop_frame = frame[:crop_y, :, :]

    # YOLO segmentation 推論
    # YOLO segmentation inference
    results = model.predict(source=crop_frame, save=False, show=False, verbose=False)
    result = results[0]

    # 如果有多條 lane mask，只保留最靠近車輛中心的
    # If there are multiple lane masks, keep only the one closest to the vehicle center
    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data
        
        # 計算每條 mask 的水平中心
        # Calculate the horizontal center of each mask
        cx_list = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                cx_list.append(np.inf)
            else:
                cx_list.append(np.mean(xs))

        # 找最靠近車輛中心的 mask
        # Find the mask closest to the vehicle center
        best_idx = np.argmin(np.abs(np.array(cx_list) - vehicle_center_x))
        best_mask = masks[best_idx]
        best_box = result.boxes.data[best_idx]

        # === 關鍵修改：只保留最中心的遮罩和邊界框 ===
        # === Key modification: Keep only the center mask and its bounding box ===
        
        # 只保留這條 mask
        # Keep only this mask
        result.masks.data = torch.from_numpy(np.expand_dims(best_mask, axis=0))

        # 只保留這個邊界框
        # Keep only this bounding box
        result.boxes.data = torch.from_numpy(np.expand_dims(best_box.cpu().numpy(), axis=0))

    # 繪製 mask 與 bounding box
    # Draw the mask and bounding box
    crop_with_mask = result.plot()

    # 疊回完整影像
    # Overlay the cropped image with the mask back onto the full image
    img_with_mask = frame.copy()
    img_with_mask[:crop_y, :, :] = crop_with_mask

    # === 在裁切區域中心畫黃色三角形 ===
    # Draw a yellow triangle at the center of the cropped area
    center_x = width // 2 + triangle_offset
    center_y = crop_y
    triangle_size = 20

    pts = np.array([
        [center_x, center_y - triangle_size],      # 上頂點 / Top vertex
        [center_x - triangle_size, center_y + triangle_size], # 左下 / Bottom left
        [center_x + triangle_size, center_y + triangle_size]   # 右下 / Bottom right
    ], np.int32)

    cv2.fillPoly(img_with_mask, [pts], (0, 255, 255))  # 黃色 (BGR) / Yellow (BGR)

    # 寫入影片
    # Write to video file
    out.write(img_with_mask)

    # 顯示影像
    # Display the image
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection", 960, 540)
    cv2.imshow("Lane Detection", img_with_mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cv2.destroyAllWindows()
