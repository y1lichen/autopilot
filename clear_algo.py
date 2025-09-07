import cv2
import numpy as np
import os
from collections import deque


# 新增參數：裁切掉畫面的底部百分比
# New parameter: Percentage of the bottom of the frame to crop
CROP_BOTTOM_PERCENTAGE = 0.35
CROP_LEFT_PERCENTAGE = 0.35
CROP_RIGHT_PERCENTAGE = 0.1

# 🔍 1. 多層次的車道線偵測策略
def preprocess_frame(frame):
    """
    結合顏色與梯度二值化，增強在不同光線條件下的穩健性。
    Returns a combined binary image.
    """
    # ✅ 顏色與梯度二值化結合
    # HLS 色彩空間用於偵測白色與黃色線條，對光線變化更不敏感
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
    yellow_mask = cv2.inRange(hls, np.array([15, 30, 115]), np.array([35, 204, 255]))
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Sobel 梯度偵測，用於找出所有高對比度的邊緣
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ✅ 新增：使用 CLAHE 提升暗處對比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_gray = clahe.apply(gray)
    
    sobelx = cv2.Sobel(clahe_gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    sobel_mask = cv2.inRange(np.uint8(255 * abs_sobelx / np.max(abs_sobelx)), 50, 255)
    
    # ✅ 新增：基於亮度通道的二值化，對暗處特別有效
    hls_l = hls[:, :, 1]
    ret, light_mask = cv2.threshold(hls_l, 200, 255, cv2.THRESH_BINARY)
    
    # 結合所有遮罩
    combined_mask = cv2.bitwise_or(color_mask, sobel_mask)
    combined_mask = cv2.bitwise_or(combined_mask, light_mask)
    return combined_mask

def estimate_turn(left_fit, right_fit):
    if left_fit is None or right_fit is None:
        return "Detecting..."
    
    y_eval = 472
    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

    delta = right_x - left_x - 300 # 假設的平均車道寬度為300像素
    if abs(delta) < 50:
        return "Straight"
    elif delta < 0:
        return "Turning Right"
    else:
        return "Turning Left"

# ==== Main Loop ====
frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")]
)

if not frame_files:
    raise RuntimeError(f"No frames found in {frames_dir}")

# 讀第一張確定大小
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]
crop_height = int(height * (1 - CROP_BOTTOM_PERCENTAGE))
crop_width = int(width * (1 - CROP_LEFT_PERCENTAGE - CROP_RIGHT_PERCENTAGE))
start_x = int(width * CROP_LEFT_PERCENTAGE)
end_x = int(width * (1 - CROP_RIGHT_PERCENTAGE))
# 設定輸出影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_bird_eye_view.mp4", fourcc, 20.0, (crop_width, crop_height))

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue
    
    # 根據 CROP_BOTTOM_PERCENTAGE 裁切畫面
    frame = frame[:crop_height, start_x:end_x]
    frame = preprocess_frame(frame)

    cv2.namedWindow("ADAS Lane Detection - Bird's Eye View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS Lane Detection - Bird's Eye View", 960, 540)
    cv2.imshow("ADAS Lane Detection - Bird's Eye View", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
