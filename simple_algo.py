import cv2
import numpy as np
import os

# ====== 參數設定 ======
CROP_BOTTOM_PERCENTAGE = 0.4
CROP_LEFT_PERCENTAGE = 0.35
CROP_RIGHT_PERCENTAGE = 0.2
WINDOW_WIDTH = 20
WINDOW_HEIGHT = 50
MIN_LANE_POINTS = 5
OFFSET_THRESHOLD = 20
HIST_THRESHOLD = 75  # histogram 高度閾值

# 車頭三角形參數
CAR_TRIANGLE_Y = 0.6    # 三角形中心的高度 (0~1, 0=頂部, 1=底部)
CAR_TRIANGLE_OFFSET = 50  # 左右偏移，單位: 像素
CAR_TRIANGLE_SIZE = 60    # 正三角形邊長 (固定大小)

# ====== 前處理 ======
def preprocess_frame(frame):
    """
    對輸入的畫面進行前處理，以更好地辨識車道線。
    包括色彩空間轉換、遮罩、Sobel邊緣偵測和形態學操作。
    """
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, np.array([0, 120, 0]), np.array([255, 255, 255]))
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

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50
    mask_filtered = np.zeros_like(combined_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(mask_filtered, [cnt], -1, 255, -1)

    return mask_filtered

# ====== Bird's Eye View ======
def apply_bird_eye_view(frame):
    """將圖像轉換為鳥瞰圖。"""
    height, width = frame.shape[:2]
    src = np.float32([
        [width * 0.3, height * 0.7],
        [width * 0.45, height * 0.7],
        [width * 0.35, height * 1.0],
        [width * 0.2, height * 1.0]
    ])
    dst = np.float32([
        [width * 0.3, 0],
        [width * 0.5, 0],
        [width * 0.4, height],
        [width * 0.3, height]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    bird_eye = cv2.warpPerspective(frame, M, (width, height))
    return bird_eye, M, Minv

# ====== 滑動窗口追蹤車道線 ======
def sliding_window_lane_detection(binary_warped, car_x_bev, prevLx=[], prevRx=[]):
    """
    使用滑動視窗在二值化鳥瞰圖上尋找車道線。
    並在圖像上繪製這些滑動視窗。
    """
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]*0.75):, :], axis=0)
    
    # 創建一個彩色圖像副本用於繪製
    window_drawing_img = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

    left_hist = histogram[:int(car_x_bev)]
    right_hist = histogram[int(car_x_bev):]

    left_base = np.argmax(left_hist) if np.max(left_hist) > HIST_THRESHOLD else int(car_x_bev//2)
    right_base = np.argmax(right_hist) + int(car_x_bev) if np.max(right_hist) > HIST_THRESHOLD else int((binary_warped.shape[1]+car_x_bev)//2)

    lx_pts, rx_pts = [], []
    y = binary_warped.shape[0]

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
        cv2.rectangle(window_drawing_img, (x_start, y), (x_end, y-WINDOW_HEIGHT), (0,255,0), 2)

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
        cv2.rectangle(window_drawing_img, (x_start, y), (x_end, y-WINDOW_HEIGHT), (0,0,255), 2)

        y -= WINDOW_HEIGHT

    if len(lx_pts) == 0: lx_pts = prevLx
    else: prevLx = lx_pts
    if len(rx_pts) == 0: rx_pts = prevRx
    else: prevRx = rx_pts

    return lx_pts, rx_pts, window_drawing_img, prevLx, prevRx, histogram

# ====== 畫車頭正三角形 (固定大小) ======
def draw_car_triangle(frame, center_x=None, center_y=None):
    """在圖像上繪製代表車頭的三角形。"""
    h, w = frame.shape[:2]
    if center_x is None:
        center_x = w // 2 + CAR_TRIANGLE_OFFSET
    if center_y is None:
        center_y = int(h * CAR_TRIANGLE_Y)

    height_tri = int(np.sqrt(3)/2 * CAR_TRIANGLE_SIZE)

    pts = np.array([
        [center_x, center_y - 2*height_tri//3],
        [center_x - CAR_TRIANGLE_SIZE//2, center_y + height_tri//3],
        [center_x + CAR_TRIANGLE_SIZE//2, center_y + height_tri//3]
    ], np.int32).reshape((-1,1,2))

    cv2.fillPoly(frame, [pts], (0,255,255))
    return frame

# ====== 畫車道線到原圖 ======
def draw_lane_on_full_frame(full_frame, left_pts, right_pts, Minv, crop_offset_y, crop_offset_x):
    """將檢測到的車道線從鳥瞰圖轉換並繪製到原始圖像上。"""
    overlay = full_frame.copy()
    if len(left_pts) > 0 and len(right_pts) > 0:
        left = np.array(left_pts, dtype=np.float32).reshape(-1,1,2)
        right = np.array(right_pts, dtype=np.float32).reshape(-1,1,2)

        left_unwarp = cv2.perspectiveTransform(left, Minv)
        right_unwarp = cv2.perspectiveTransform(right, Minv)

        left_unwarp[:,:,0] += crop_offset_x
        right_unwarp[:,:,0] += crop_offset_x

        cv2.polylines(overlay, [np.int32(left_unwarp)], False, (0,255,0), 5)
        cv2.polylines(overlay, [np.int32(right_unwarp)], False, (0,255,0), 5)

        pts = np.vstack((left_unwarp, right_unwarp[::-1]))
        cv2.fillPoly(overlay, [np.int32(pts)], (255,0,0))
        overlay = cv2.addWeighted(overlay, 0.3, full_frame, 0.7, 0)
    return overlay

# ====== 在 Bird Eye 上畫車道線 ======
def draw_lane_on_bird_eye(bird_eye_color, left_pts, right_pts, car_bev_x=None, car_bev_y=None):
    """在鳥瞰圖上繪製車道線和車頭三角形。"""
    bird_eye_with_lane = bird_eye_color.copy()

    # 先畫車頭三角形在 Bird Eye View 上
    if car_bev_x is not None and car_bev_y is not None:
        draw_car_triangle(bird_eye_with_lane, int(car_bev_x), int(car_bev_y))
    
    if len(left_pts) > 0 and len(right_pts) > 0:
        cv2.polylines(bird_eye_with_lane, [np.int32(left_pts)], False, (0, 255, 0), 5)
        cv2.polylines(bird_eye_with_lane, [np.int32(right_pts)], False, (0, 255, 0), 5)

        pts = np.vstack((np.int32(left_pts), np.int32(right_pts[::-1])))
        overlay = bird_eye_with_lane.copy()
        cv2.fillPoly(overlay, [pts], (255, 0, 0))
        bird_eye_with_lane = cv2.addWeighted(overlay, 0.3, bird_eye_with_lane, 0.7, 0)
    
    return bird_eye_with_lane

# ====== 畫 Histogram ======
def draw_histogram_on_bird_eye(bird_eye_color, histogram):
    """將直方圖繪製在鳥瞰圖下方。"""
    height, width = bird_eye_color.shape[:2]
    hist_img = np.zeros((150, width, 3), dtype=np.uint8)
    if np.max(histogram) > 0:
        norm_hist = (histogram / np.max(histogram) * hist_img.shape[0]).astype(np.int32)
    else:
        norm_hist = histogram
    for x, h in enumerate(norm_hist):
        cv2.line(hist_img, (x, hist_img.shape[0]), (x, hist_img.shape[0]-h), (0,255,0), 1)
    combined = np.vstack((bird_eye_color, hist_img))
    return combined

# ====== 判斷方向 ======
def detect_turn(left_pts, right_pts, frame_width):
    """根據車道線的形狀和位置判斷行駛方向。"""
    if len(left_pts) < MIN_LANE_POINTS or len(right_pts) < MIN_LANE_POINTS:
        return "detecting"
    left_fit = np.polyfit([p[1] for p in left_pts], [p[0] for p in left_pts], 2)
    right_fit = np.polyfit([p[1] for p in right_pts], [p[0] for p in right_pts], 2)
    a_left, a_right = left_fit[0], right_fit[0]
    if abs(a_left) < 1e-5 and abs(a_right) < 1e-5:
        curve = "straight"
    elif a_left < 0 and a_right < 0:
        curve = "turn left"
    elif a_left > 0 and a_right > 0:
        curve = "turn right"
    else:
        curve = "detecting"
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
out = cv2.VideoWriter("output_lane_turn_detection.mp4", fourcc, 20.0, (width, height))
out_bird_eye = cv2.VideoWriter("output_bird_eye_lane.mp4", fourcc, 20.0, (crop_width, crop_height+150))

# ====== 視窗設定 960x540 ======
cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Lane Detection", 960, 540)
cv2.namedWindow("Bird's Eye View with Lane + Histogram", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bird's Eye View with Lane + Histogram", 960, 540)
# 這個視窗將顯示帶有滑動視窗矩形的黑白鳥瞰圖
cv2.namedWindow("Bird's Eye View with Sliding Windows", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bird's Eye View with Sliding Windows", 960, 540)

prevLx, prevRx = [], []
for fname in frame_files:
    full_frame = cv2.imread(os.path.join(frames_dir, fname))
    if full_frame is None:
        continue

    # crop
    frame_crop = full_frame[:crop_height, start_x:end_x]
    bird_eye_frame_color, M, Minv = apply_bird_eye_view(frame_crop)
    processed_frame = preprocess_frame(bird_eye_frame_color)

    # 原圖車頭三角形中心點 (考慮 crop)
    car_center_x_crop = (width//2 + CAR_TRIANGLE_OFFSET) - start_x
    car_center_y_crop = int(frame_crop.shape[0] * CAR_TRIANGLE_Y)
    pts = np.array([[[car_center_x_crop, car_center_y_crop]]], dtype=np.float32)
    car_bev = cv2.perspectiveTransform(pts, M)
    car_bev_x, car_bev_y = car_bev[0,0,0], car_bev[0,0,1]

    # 這裡的 window_drawing_img 就是你要求的回傳圖片
    lx_pts, rx_pts, window_drawing_img, prevLx, prevRx, histogram = sliding_window_lane_detection(processed_frame, car_bev_x, prevLx, prevRx)

    # 畫在原圖
    frame_with_lane_full = draw_lane_on_full_frame(full_frame.copy(), lx_pts, rx_pts, Minv, 0, start_x)
    frame_with_lane_full = draw_car_triangle(frame_with_lane_full)

    # Bird Eye View 畫車道線 + 車頭
    bird_eye_with_lane = draw_lane_on_bird_eye(bird_eye_frame_color.copy(), lx_pts, rx_pts, car_bev_x, car_bev_y)
    bird_eye_with_hist = draw_histogram_on_bird_eye(bird_eye_with_lane, histogram)

    turn_status = detect_turn(lx_pts, rx_pts, frame_crop.shape[1])
    cv2.putText(frame_with_lane_full, turn_status, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    cv2.putText(bird_eye_with_hist, turn_status, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Lane Detection", frame_with_lane_full)
    out.write(frame_with_lane_full)

    cv2.imshow("Bird's Eye View with Lane + Histogram", bird_eye_with_hist)
    out_bird_eye.write(bird_eye_with_hist)

    # 顯示帶有滑動視窗的鳥瞰圖
    cv2.imshow("Bird's Eye View with Sliding Windows", window_drawing_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
out_bird_eye.release()
cv2.destroyAllWindows()
