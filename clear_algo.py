import cv2
import numpy as np
import os
from collections import deque

# ====== 參數設定 ======
CROP_BOTTOM_PERCENTAGE = 0.35
CROP_LEFT_PERCENTAGE = 0.35
CROP_RIGHT_PERCENTAGE = 0.1


# ====== Bird's Eye View 轉換 ======
def warp_perspective(frame):
    """
    對輸入影像做透視轉換 (Bird's Eye View)
    回傳轉換後影像
    """
    height, width = frame.shape[:2]

    # 定義來源點 (依照相機視角需要調整)
    src = np.float32([
        [width * 0.35, height * 0.55],   # 左上
        [width * 0.45, height * 0.55],   # 右上
        [width * 0.8,  height * 1.0],   # 右下
        [width * 0.1,  height * 1.0]    # 左下
    ])

    # 定義目標點 (轉成矩形)
    dst = np.float32([
        [width * 0.4, 0],     # 左上
        [width * 1.0, 0],     # 右上
        [width * 1.0, height],# 右下
        [width * 0.4, height] # 左下
    ])

    # 計算透視矩陣
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (width, height))

    return warped


# ====== 前處理：顏色 + 梯度 ======
def preprocess_frame(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
    yellow_mask = cv2.inRange(hls, np.array([15, 30, 115]), np.array([35, 204, 255]))
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)

    sobelx = cv2.Sobel(clahe_gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    sobel_mask = cv2.inRange(np.uint8(255 * abs_sobelx / np.max(abs_sobelx)), 50, 255)

    hls_l = hls[:, :, 1]
    _, light_mask = cv2.threshold(hls_l, 200, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_or(color_mask, sobel_mask)
    combined_mask = cv2.bitwise_or(combined_mask, light_mask)
    return combined_mask


# ====== 判斷方向 ======
def estimate_turn(left_fit, right_fit):
    if left_fit is None or right_fit is None:
        return "Detecting..."

    y_eval = 472
    left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]

    delta = right_x - left_x - 300  # 假設的平均車道寬度為 300 像素
    if abs(delta) < 50:
        return "Straight"
    elif delta < 0:
        return "Turning Right"
    else:
        return "Turning Left"


# ====== Main ======
frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")]
)

if not frame_files:
    raise RuntimeError(f"No frames found in {frames_dir}")

first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]
crop_height = int(height * (1 - CROP_BOTTOM_PERCENTAGE))
crop_width = int(width * (1 - CROP_LEFT_PERCENTAGE - CROP_RIGHT_PERCENTAGE))
start_x = int(width * CROP_LEFT_PERCENTAGE)
end_x = int(width * (1 - CROP_RIGHT_PERCENTAGE))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_bird_eye_view.mp4", fourcc, 20.0, (crop_width, crop_height))

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue

    # 裁切
    frame = frame[:crop_height, start_x:end_x]

    # Bird's Eye View
    bird_view = warp_perspective(frame)

    # 前處理
    processed = preprocess_frame(bird_view)

    # 顯示
    cv2.namedWindow("ADAS Lane Detection - Bird's Eye View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS Lane Detection - Bird's Eye View", 960, 540)
    cv2.imshow("ADAS Lane Detection - Bird's Eye View", processed)

    out.write(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
