import cv2
import numpy as np
import os

# ====== 參數設定 ======
CROP_BOTTOM_PERCENTAGE = 0.35  # 裁掉底部比例
CROP_LEFT_PERCENTAGE = 0.35    # 裁掉左側比例
CROP_RIGHT_PERCENTAGE = 0.1    # 裁掉右側比例
WINDOW_WIDTH = 100              # 滑動視窗寬度
WINDOW_HEIGHT = 50              # 滑動視窗高度
MIN_LANE_POINTS = 5             # 擬合多項式所需最少點數
OFFSET_THRESHOLD = 20           # 車道中心偏移判斷閾值 (像素)

# ====== 前處理 ======
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
    height, width = frame.shape[:2]
    src = np.float32([
        [width * 0.15, height * 0.75],
        [width * 0.5, height * 0.75],
        [width * 0.65, height * 1.0],
        [width * 0.00, height * 1.0]
    ])
    dst = np.float32([
        [width * 0.1, 0],
        [width * 0.95, 0],
        [width * 0.95, height],
        [width * 0.1, height]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)   # 反轉換矩陣
    bird_eye = cv2.warpPerspective(frame, M, (width, height))
    return bird_eye, M, Minv

# ====== 滑動窗口追蹤車道線 ======
def sliding_window_lane_detection(binary_warped, prevLx=[], prevRx=[]):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    lx_pts, rx_pts = [], []
    y = binary_warped.shape[0]
    mask_copy = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

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
                lx_pts.append((left_base, y - WINDOW_HEIGHT//2))
        cv2.rectangle(mask_copy, (x_start, y), (x_end, y-WINDOW_HEIGHT), (0,255,0), 2)

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
                rx_pts.append((right_base, y - WINDOW_HEIGHT//2))
        cv2.rectangle(mask_copy, (x_start, y), (x_end, y-WINDOW_HEIGHT), (0,0,255), 2)

        y -= WINDOW_HEIGHT

    # 空值補齊
    if len(lx_pts) == 0: lx_pts = prevLx
    else: prevLx = lx_pts
    if len(rx_pts) == 0: rx_pts = prevRx
    else: prevRx = rx_pts

    return lx_pts, rx_pts, mask_copy, prevLx, prevRx

# ====== 畫回原圖並加透明區塊 ======
def draw_lane_on_original(frame, left_pts, right_pts, Minv):
    overlay = frame.copy()
    if len(left_pts) > 0 and len(right_pts) > 0:
        left = np.array(left_pts, dtype=np.float32).reshape(-1,1,2)
        right = np.array(right_pts, dtype=np.float32).reshape(-1,1,2)

        left_unwarp = cv2.perspectiveTransform(left, Minv)
        right_unwarp = cv2.perspectiveTransform(right, Minv)

        # 畫車道邊界
        cv2.polylines(overlay, [np.int32(left_unwarp)], False, (0,255,0), 5)
        cv2.polylines(overlay, [np.int32(right_unwarp)], False, (0,255,0), 5)

        # 填滿區域 (透明藍色)
        pts = np.vstack((left_unwarp, right_unwarp[::-1]))
        cv2.fillPoly(overlay, [np.int32(pts)], (255,0,0))
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    return frame

# ====== 判斷方向 ======
def detect_turn(left_pts, right_pts, frame_width):
    if len(left_pts) < MIN_LANE_POINTS or len(right_pts) < MIN_LANE_POINTS:
        return "detecting"

    # 擬合二次多項式
    left_fit = np.polyfit([p[1] for p in left_pts], [p[0] for p in left_pts], 2)
    right_fit = np.polyfit([p[1] for p in right_pts], [p[0] for p in right_pts], 2)
    a_left, a_right = left_fit[0], right_fit[0]

    # 判斷曲率
    if abs(a_left) < 1e-5 and abs(a_right) < 1e-5:
        curve = "straight"
    elif a_left < 0 and a_right < 0:
        curve = "turn left"
    elif a_left > 0 and a_right > 0:
        curve = "turn right"
    else:
        curve = "detecting"

    # 車道中心判斷微調
    lane_center_x = np.mean([left_pts[-1][0], right_pts[-1][0]])
    vehicle_center_x = frame_width / 2
    offset = lane_center_x - vehicle_center_x

    if curve == "straight":
        if offset > OFFSET_THRESHOLD:
            turn = "adjust left"
        elif offset < -OFFSET_THRESHOLD:
            turn = "adjust right"
        else:
            turn = "keep straight"
    elif curve in ["turn left", "turn right"]:
        turn = curve
    else:
        turn = "detecting"

    return turn

# ====== Main ======
frames_dir = "dataset/run_1755702281/frames"
frames_dir = "dataset/run_1755702912/frames" # night
frames_dir = "dataset/run_1756133797/frames"
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
out = cv2.VideoWriter("output_lane_turn_detection.mp4", fourcc, 20.0, (crop_width, crop_height))

prevLx, prevRx = [], []

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue

    frame_crop = frame[:crop_height, start_x:end_x]
    bird_eye_frame, M, Minv = apply_bird_eye_view(frame_crop)
    processed_frame = preprocess_frame(bird_eye_frame)

    lx_pts, rx_pts, mask_copy, prevLx, prevRx = sliding_window_lane_detection(processed_frame, prevLx, prevRx)

    frame_with_lane = draw_lane_on_original(frame_crop.copy(), lx_pts, rx_pts, Minv)

    # 判斷方向
    turn_status = detect_turn(lx_pts, rx_pts, frame_crop.shape[1])

    # 顯示方向文字
    cv2.putText(frame_with_lane, turn_status, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection", 960, 540)
    cv2.imshow("Lane Detection", frame_with_lane)
    out.write(frame_with_lane)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
