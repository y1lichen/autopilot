import cv2
import numpy as np
import os

# ====== 參數設定 ======
CROP_BOTTOM_PERCENTAGE = 0.42
CROP_LEFT_PERCENTAGE = 0.35
CROP_RIGHT_PERCENTAGE = 0.1
WINDOW_WIDTH = 100
WINDOW_HEIGHT = 40

# ====== 前處理 ======
def preprocess_frame(frame):
    """
    對影像進行前處理以增強車道線特徵。
    """
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

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 過濾小區域
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50
    mask_filtered = np.zeros_like(combined_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(mask_filtered, [cnt], -1, 255, -1)

    return mask_filtered

# ====== Bird's Eye View ======
def apply_bird_eye_view(frame):
    """
    將影像轉換為具有扇形視角的鳥瞰圖。
    """
    height, width = frame.shape[:2]
    # 來源點 (src): 保持原有設定，定義原始影像中感興趣的梯形區域
    src = np.float32(
    [
        [width * 0.4, height * 0.6],
        [width * 0.6, height * 0.6],
        [width * 0.7, height * 1.0],
        [width * 0.3, height * 1.0],
    ])
    # 目標點 (dst): 定義轉換後影像中的扇形區域，近處較窄，遠處較寬
    dst = np.float32(
    [
        [width * 0.4, 0],
        [width * 0.6, 0],
        [width * 0.6, height],
        [width * 0.4, height]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    bird_eye = cv2.warpPerspective(frame, M, (width, height))
    return bird_eye, src, dst

# ====== 滑動窗口追蹤車道線 Function ======
def sliding_window_lane_detection(binary_warped, prevLx=[], prevRx=[]):
    """
    使用滑動窗口法追蹤車道線。
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    lx, rx = [], []
    y = binary_warped.shape[0]
    mask_copy = binary_warped.copy()

    while y > 0:
        # 左邊
        x_start = max(left_base - WINDOW_WIDTH//2, 0)
        x_end = min(left_base + WINDOW_WIDTH//2, binary_warped.shape[1])
        window_left = binary_warped[y-WINDOW_HEIGHT:y, x_start:x_end]
        contours, _ = cv2.findContours(window_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                left_base = x_start + cx
                lx.append(left_base)
        cv2.rectangle(mask_copy, (x_start, y), (x_end, y-WINDOW_HEIGHT), 255, 2)

        # 右邊
        x_start = max(right_base - WINDOW_WIDTH//2, 0)
        x_end = min(right_base + WINDOW_WIDTH//2, binary_warped.shape[1])
        window_right = binary_warped[y-WINDOW_HEIGHT:y, x_start:x_end]
        contours, _ = cv2.findContours(window_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                right_base = x_start + cx
                rx.append(right_base)
        cv2.rectangle(mask_copy, (x_start, y), (x_end, y-WINDOW_HEIGHT), 255, 2)

        y -= WINDOW_HEIGHT

    # 空值補齊
    if len(lx) == 0: lx = prevLx
    else: prevLx = lx
    if len(rx) == 0: rx = prevRx
    else: prevRx = rx

    return lx, rx, mask_copy, prevLx, prevRx

# ====== Main ======
frames_dir = "dataset/run_1756133797/frames"
# frames_dir = "dataset/run_1755702281/frames"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])
if not frame_files:
    raise RuntimeError("No frames found!")

first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]
crop_height = int(height * (1 - CROP_BOTTOM_PERCENTAGE))
crop_width = int(width * (1 - CROP_LEFT_PERCENTAGE - CROP_RIGHT_PERCENTAGE))
start_x = int(width * CROP_LEFT_PERCENTAGE)
end_x = int(width * (1 - CROP_RIGHT_PERCENTAGE))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_bird_eye_view_sliding_window.mp4", fourcc, 20.0, (crop_width, crop_height))

prevLx, prevRx = [], []

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue

    frame = frame[:crop_height, start_x:end_x]
    bird_eye_frame, src_points, dst_points = apply_bird_eye_view(frame)
    # processed_frame = preprocess_frame(bird_eye_frame)

    # lx, rx, mask_copy, prevLx, prevRx = sliding_window_lane_detection(processed_frame, prevLx, prevRx)

    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

    # 設定視窗大小
    cv2.resizeWindow("Lane Detection", 960, 540)  # 寬 960 高 540

    # 顯示影像
    cv2.imshow("Lane Detection", bird_eye_frame)
    out.write(bird_eye_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
