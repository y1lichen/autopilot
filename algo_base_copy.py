import cv2
import numpy as np
import os
from collections import deque

left_history = deque(maxlen=10)
right_history = deque(maxlen=10)

def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([[
        (0, height),
        (width, height),
        (width // 2 + 200, height // 5),
        (width // 2 - 200, height // 5)
    ]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if slope == 0: slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    img_center = image.shape[1] // 2
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0: continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < -0.3 and x1 < img_center and x2 < img_center:
            left_fit.append((slope, intercept))
        elif slope > 0.3 and x1 > img_center and x2 > img_center:
            right_fit.append((slope, intercept))
    left_line = make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None
    return left_line, right_line

def draw_lane_area(image, left_line, right_line):
    lane_image = np.zeros_like(image)
    if left_line is not None:
        cv2.line(lane_image, tuple(left_line[:2]), tuple(left_line[2:]), (255, 0, 255), 6)
    if right_line is not None:
        cv2.line(lane_image, tuple(right_line[:2]), tuple(right_line[2:]), (255, 0, 255), 6)
    if left_line is not None and right_line is not None:
        if left_line[0] < right_line[0] and left_line[2] < right_line[2]:
            points = np.array([[
                tuple(left_line[:2]),
                tuple(left_line[2:]),
                tuple(right_line[2:]),
                tuple(right_line[:2])
            ]], dtype=np.int32)
            cv2.fillPoly(lane_image, points, (0, 255, 0))
    return lane_image

def estimate_turn(left, right):
    if left is None or right is None:
        return "Detecting..."
    mid_bottom = (left[0] + right[0]) // 2
    mid_top = (left[2] + right[2]) // 2
    delta = mid_top - mid_bottom
    if abs(delta) < 30:
        return "Straight"
    elif delta < 0:
        return "Turning Left"
    else:
        return "Turning Right"


# ==== 改這裡：讀取 run_* 的 frames ====
frames_dir = "dataset/run_1756133797/frames"   # 修改成你的 run_* 資料夾
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")]
)

if not frame_files:
    raise RuntimeError(f"No frames found in {frames_dir}")

# 讀第一張確定大小
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]

# 設定輸出影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_lane_from_frames.mp4", fourcc, 20.0, (width, height))

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue

    # --- 和你原本程式一樣的 lane detection 流程 ---
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
    yellow = cv2.inRange(hls, np.array([15, 30, 115]), np.array([35, 204, 255]))
    mask = cv2.bitwise_or(white, yellow)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 60,
                            minLineLength=40, maxLineGap=150)

    left_line, right_line = average_slope_intercept(frame, lines)

    if left_line is not None:
        left_history.append(left_line)
    if right_line is not None:
        right_history.append(right_line)

    left_avg = np.mean(left_history, axis=0).astype(int) if left_history else None
    right_avg = np.mean(right_history, axis=0).astype(int) if right_history else None

    overlay = draw_lane_area(frame, left_avg, right_avg)
    # output = cv2.addWeighted(frame, 0.8, overlay, 1, 1)

    # direction = estimate_turn(left_avg, right_avg)
    # cv2.putText(output, direction, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # out.write(output)
    # cv2.imshow("ADAS Lane Detection from Frames", output)
    output = cv2.addWeighted(frame, 0.8, overlay, 1, 1)

    direction = estimate_turn(left_avg, right_avg)
    cv2.putText(output, direction, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # --- 這裡開始是修改的程式碼 ---
    # 建立一個和 output 圖像一樣大小的黑色背景
    roi_mask_color = np.zeros_like(output)

    # 使用您原本定義的多邊形來建立 ROI 遮罩
    height, width = output.shape[:2]
    polygons = np.array([[
        (0, height




            ),
        (width, height),
        (width // 2 + 200, height // 5),
        (width // 2 - 200, height // 5)
    ]])
    cv2.fillPoly(roi_mask_color, polygons, (255, 255, 255))

    # 將 output 圖像與 ROI 遮罩進行位元運算，只保留 ROI 內的內容
    output_roi_only = cv2.bitwise_and(output, roi_mask_color)

    # --- 這裡結束是修改的程式碼 ---
    
    out.write(output_roi_only)
    cv2.imshow("ADAS Lane Detection from Frames", output_roi_only)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
