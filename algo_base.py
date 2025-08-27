import cv2
import numpy as np
from collections import deque

# 保存歷史檢測的車道線座標，避免單幀抖動
left_history = deque(maxlen=10)
right_history = deque(maxlen=10)


def region_of_interest(img):
    """
    定義影像中「感興趣區域 (ROI)」，只保留前方道路三角形區域。
    避免天空、車子邊緣等干擾。
    """
    height, width = img.shape[:2]
    polygons = np.array([[
        (100, height),               # 左下角
        (width - 100, height),       # 右下角
        (width // 2 + 90, height // 2 + 40),   # 右上
        (width // 2 - 90, height // 2 + 40)    # 左上
    ]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)  # 填滿 ROI 多邊形
    return cv2.bitwise_and(img, mask)  # 只保留 ROI 內部的影像


def make_coordinates(image, line_params):
    """
    根據斜率 slope 和截距 intercept 計算車道線的兩個端點座標。
    """
    slope, intercept = line_params
    y1 = image.shape[0]          # 底部（畫面最下方）
    y2 = int(y1 * 0.6)           # 上方某一高度 (約 60%)
    if slope == 0: slope = 0.1   # 避免除以 0
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    """
    將 Hough Transform 找到的線段，分為左線與右線，
    分別取平均，得到更穩定的左右車道線。
    """
    left_fit, right_fit = [], []
    img_center = image.shape[1] // 2

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0: continue  # 避免垂直線除以零
        slope = (y2 - y1) / (x2 - x1)   # 斜率
        intercept = y1 - slope * x1     # 截距

        # 左車道：斜率負且在線中心左側
        if slope < -0.3 and x1 < img_center and x2 < img_center:
            left_fit.append((slope, intercept))
        # 右車道：斜率正且在線中心右側
        elif slope > 0.3 and x1 > img_center and x2 > img_center:
            right_fit.append((slope, intercept))

    left_line = make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None

    return left_line, right_line


def draw_lane_area(image, left_line, right_line):
    """
    在畫面上畫出左右車道線，以及中間的綠色區域。
    """
    lane_image = np.zeros_like(image)

    # 畫左右線
    if left_line is not None:
        cv2.line(lane_image, tuple(left_line[:2]), tuple(left_line[2:]), (255, 0, 255), 6)
    if right_line is not None:
        cv2.line(lane_image, tuple(right_line[:2]), tuple(right_line[2:]), (255, 0, 255), 6)

    # 畫填滿的車道區域
    if left_line is not None and right_line is not None:
        if left_line[0] < right_line[0] and left_line[2] < right_line[2]:
            points = np.array([[
                tuple(left_line[:2]),
                tuple(left_line[2:]),
                tuple(right_line[2:]),
                tuple(right_line[:2])
            ]], dtype=np.int32)
            cv2.fillPoly(lane_image, points, (0, 255, 0))  # 綠色填充

    return lane_image


def estimate_turn(left, right):
    """
    根據左右車道線的平均位置，推測車輛方向：
    - Straight: 直行
    - Turning Left: 左轉
    - Turning Right: 右轉
    """
    if left is None or right is None:
        return "Detecting..."

    # 車道線底部與上方中點
    mid_bottom = (left[0] + right[0]) // 2
    mid_top = (left[2] + right[2]) // 2
    delta = mid_top - mid_bottom

    if abs(delta) < 30:
        return "Straight"
    elif delta < 0:
        return "Turning Left"
    else:
        return "Turning Right"


# ===== 主程式開始 =====
video_path = "Sample_video"  # 輸入影片檔路徑
cap = cv2.VideoCapture(video_path)

# 定義影片輸出編碼器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_lane_stable.mp4", fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ===== 影像處理流程 =====

    # 1. 轉換顏色空間 → HLS（較容易分離白色、黃色）
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
    yellow = cv2.inRange(hls, np.array([15, 30, 115]), np.array([35, 204, 255]))
    mask = cv2.bitwise_or(white, yellow)             # 白線 + 黃線遮罩
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # 2. 灰階、模糊、邊緣檢測
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)

    # 3. 限制感興趣區域 (ROI)
    cropped = region_of_interest(edges)

    # 4. Hough Transform 找出直線
    lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 60,
                            minLineLength=40, maxLineGap=150)

    # 5. 計算左右車道線
    left_line, right_line = average_slope_intercept(frame, lines)

    # 6. 保存歷史結果 → 平滑化
    if left_line is not None:
        left_history.append(left_line)
    if right_line is not None:
        right_history.append(right_line)

    left_avg = np.mean(left_history, axis=0).astype(int) if left_history else None
    right_avg = np.mean(right_history, axis=0).astype(int) if right_history else None

    # 7. 繪製車道線與區域
    overlay = draw_lane_area(frame, left_avg, right_avg)
    output = cv2.addWeighted(frame, 0.8, overlay, 1, 1)

    # 8. 推測車輛方向
    direction = estimate_turn(left_avg, right_avg)
    cv2.putText(output, direction, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # ===== 輸出影像 =====
    out.write(output)                  # 存到影片
    cv2.imshow("ADAS Lane Detection (Stable)", output)  # 顯示畫面
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 q 離開
        break

# ===== 資源釋放 =====
cap.release()
out.release()
cv2.destroyAllWindows()
