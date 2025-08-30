import cv2
import numpy as np
import os
from collections import deque

# 儲存 polyfit 係數 (a, b, c)
left_history = deque(maxlen=10)
right_history = deque(maxlen=10)

def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([[
        (int(width*0.1), height),
        (int(width*0.8), height),
        (int(width*0.6), int(height*0.5)),
        (int(width*0.4), int(height*0.5))
    ]])


    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def fit_poly(points):
    """ 給定一組 (x, y) 點，回傳二次多項式係數 (a, b, c)，對應 x = a*y^2 + b*y + c """
    if len(points) < 6:  # 點數太少不擬合
        return None
    x = points[:, 0]
    y = points[:, 1]
    return np.polyfit(y, x, 2)

def make_curve_points(image, poly_fit):
    """ 根據 poly_fit 生成車道曲線點 """
    if poly_fit is None:
        return None
    a, b, c = poly_fit
    y1 = image.shape[0]   # 底部
    y2 = int(y1 * 0.6)    # 上方 (可調整)
    curve = []
    for y in range(y1, y2, -5):  # 每 5px 取一點
        x = int(a*y*y + b*y + c)
        curve.append((x, y))
    return np.array(curve, dtype=np.int32)

def fit_lane_lines(image, lines):
    """ 從 HoughLinesP 結果分左右線，分別做 polyfit """
    if lines is None:
        return None, None
    left_points, right_points = [], []
    img_center = image.shape[1] // 2

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        # 分左右線：斜率為負 → 左線，斜率為正 → 右線
        if slope < -0.3 and x1 < img_center and x2 < img_center:
            left_points.extend([(x1, y1), (x2, y2)])
        elif slope > 0.3 and x1 > img_center and x2 > img_center:
            right_points.extend([(x1, y1), (x2, y2)])

    left_fit = fit_poly(np.array(left_points)) if left_points else None
    right_fit = fit_poly(np.array(right_points)) if right_points else None
    return left_fit, right_fit

def draw_lane_area(image, left_fit, right_fit):
    lane_image = np.zeros_like(image)

    left_curve = make_curve_points(image, left_fit)
    right_curve = make_curve_points(image, right_fit)

    if left_curve is not None:
        cv2.polylines(lane_image, [left_curve], False, (255, 0, 255), 6)
    if right_curve is not None:
        cv2.polylines(lane_image, [right_curve], False, (255, 0, 255), 6)

    if left_curve is not None and right_curve is not None:
        # 建立封閉多邊形，塗滿 lane 區域
        pts = np.vstack([left_curve, right_curve[::-1]])
        cv2.fillPoly(lane_image, [pts], (0, 255, 0))

    return lane_image

def estimate_turn(left_fit, right_fit, image):
    """ 根據 polyfit 判斷方向 """
    if left_fit is None or right_fit is None:
        return "Detecting..."

    y_eval = int(image.shape[0] * 0.7)  # 評估點 (畫面下方 70%)
    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    mid = (left_x + right_x) / 2
    center = image.shape[1] / 2

    offset = mid - center
    if abs(offset) < 30:
        return "Straight"
    elif offset < 0:
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
height = int(height * 2 / 3)
# 設定輸出影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_lane_from_frames.mp4", fourcc, 20.0, (width, height))

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue
    frame = frame[: height,:]

    # --- 車道檢測流程 ---
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
    yellow = cv2.inRange(hls, np.array([15, 30, 115]), np.array([35, 204, 255]))
    mask = cv2.bitwise_or(white, yellow)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped,
        2,
        np.pi / 180,
        100,
        minLineLength=40,
        maxLineGap=150)

    left_fit, right_fit = fit_lane_lines(frame, lines)

    # 更新歷史 (儲存 polyfit 係數)
    if left_fit is not None:
        left_history.append(left_fit)
    if right_fit is not None:
        right_history.append(right_fit)

    # 平滑處理 (取平均 polyfit 係數)
    left_avg = np.mean(left_history, axis=0) if left_history else None
    right_avg = np.mean(right_history, axis=0) if right_history else None

    overlay = draw_lane_area(frame, left_avg, right_avg)
    output = cv2.addWeighted(frame, 0.8, overlay, 1, 1)

    direction = estimate_turn(left_avg, right_avg, frame)
    cv2.putText(output, direction, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # --- ROI 遮罩 (可選) ---
    roi_mask_color = np.zeros_like(output)
    polygons = np.array([[
        (int(width*0.1), height),
        (int(width*0.8), height),
        (int(width*0.7), int(height*0.5)),
        (int(width*0.3), int(height*0.5))
    ]])



    cv2.fillPoly(roi_mask_color, polygons, (255, 255, 255))
    output_roi_only = cv2.bitwise_and(output, roi_mask_color)
    
    out.write(output_roi_only)
    cv2.imshow("ADAS Lane Detection from Frames", output_roi_only)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
