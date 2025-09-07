import cv2
import numpy as np
import os

# ====== 參數設定 ======
CROP_BOTTOM_PERCENTAGE = 0.35
CROP_LEFT_PERCENTAGE = 0.35
CROP_RIGHT_PERCENTAGE = 0.1

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

# ====== 套用 ROI 並轉換為 Bird's Eye View ======
def apply_bird_eye_view(frame):
    height, width = frame.shape[:2]

    # 定義來源點 (梯形 ROI)
    src = np.float32([
        [width * 0.15, height * 0.6],   # 左上
        [width * 0.5, height * 0.6],   # 右上
        [width * 0.65,  height * 1.0],   # 右下
        [width * 0.0,  height * 1.0]    # 左下
    ])

    # 定義目標點 (轉換成矩形)
    dst = np.float32([
        [width * 0.1, 0],
        [width * 0.95, 0],
        [width * 0.95, height],
        [width * 0.1, height]
    ])

    # 計算透視變換矩陣
    M = cv2.getPerspectiveTransform(src, dst)

    # 應用透視變換
    bird_eye = cv2.warpPerspective(frame, M, (width, height))

    return bird_eye, src, dst

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

    # Bird’s Eye View
    bird_eye_frame, src_points, dst_points = apply_bird_eye_view(frame)

    processed_frame = preprocess_frame(bird_eye_frame)

    # 顯示
    cv2.namedWindow("ADAS Lane Detection - Bird's Eye View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS Lane Detection - Bird's Eye View", 960, 540)
    cv2.imshow("ADAS Lane Detection - Bird's Eye View", bird_eye_frame)

    out.write(bird_eye_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
