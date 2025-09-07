import cv2
import numpy as np
import os
from collections import deque

# 參數設定
# Parameter Configuration
left_history = deque(maxlen=10)
right_history = deque(maxlen=10)

# 參數化 Bird's Eye View 的來源點，方便手動調校
# These points are relative to the image dimensions (width, height).
# These are the corners of the trapezoid ROI in the original image.
BEV_SRC_POINTS_REL = np.float32([
    [0.25, 0.95],
    [0.75, 0.95],
    [0.6, 0.55],
    [0.4, 0.55]
])

# 新增參數：裁切掉畫面的底部百分比
# New parameter: Percentage of the bottom of the frame to crop
CROP_BOTTOM_PERCENTAGE = 0.3

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
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    sobel_mask = cv2.inRange(np.uint8(255 * abs_sobelx / np.max(abs_sobelx)), 50, 255)

    # 結合兩種遮罩
    combined_mask = cv2.bitwise_or(color_mask, sobel_mask)
    return combined_mask

# ✅ 鳥瞰視角轉換（Bird’s Eye View）
def get_perspective_transforms(frame):
    """
    將畫面轉換為俯視視角，將曲線轉為接近直線，便於後續車道線檢測與擬合。
    取得 Bird’s Eye View (透視轉換) 的轉換矩陣。
    輸入: BGR 或灰階 frame
    輸出: 正向和反向的轉換矩陣 M 和 Minv
    """
    height, width = frame.shape[:2]

    # Use the global parameter to define the source points
    src = np.float32([
        [width * BEV_SRC_POINTS_REL[0, 0], height * BEV_SRC_POINTS_REL[0, 1]],
        [width * BEV_SRC_POINTS_REL[1, 0], height * BEV_SRC_POINTS_REL[1, 1]],
        [width * BEV_SRC_POINTS_REL[2, 0], height * BEV_SRC_POINTS_REL[2, 1]],
        [width * BEV_SRC_POINTS_REL[3, 0], height * BEV_SRC_POINTS_REL[3, 1]]
    ])
    
    # 計算目標點
    # Define the new dst points as a perfect rectangle,
    # mapping the src trapezoid to the entire output frame.
    dst = np.float32([
        [0, height],        # bottom-left
        [width, height],    # bottom-right
        [width, 0],         # top-right
        [0, 0]              # top-left
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if slope == 0: slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# 🧠 2. 智能化車道線擬合流程
# 註：此處採用 Hough 變換進行車道線擬合，但以下註解解釋了更進階的智能化擬合理念。
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
    
    # ✅ 動態修正錯誤偵測（左右線距離檢查）
    # 如果左右線間距太大或太小，會嘗試修正左線的位置，避免車道線追蹤錯誤。
    if left_line is not None and right_line is not None:
        lane_width = (right_line[0] - left_line[0])
        avg_lane_width = 300 # 假設的平均車道寬度
        if abs(lane_width - avg_lane_width) > 100:
            # 偵測到異常寬度，嘗試使用歷史資料進行修正（此處僅為邏輯示範）
            pass # 這裡可以加入更複雜的修正邏輯

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
frames_dir = "dataset/run_1756133797/frames"    # 修改成你的 run_* 資料夾
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")]
)

if not frame_files:
    raise RuntimeError(f"No frames found in {frames_dir}")

# 讀第一張確定大小
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]
crop_height = int(height * (1 - CROP_BOTTOM_PERCENTAGE))

# 設定輸出影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_bird_eye_view.mp4", fourcc, 20.0, (width, crop_height))

# 重新計算透視變換矩陣，使用裁切後的高度
M, Minv = get_perspective_transforms(first_frame[:crop_height, :])

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue
    
    # 根據 CROP_BOTTOM_PERCENTAGE 裁切畫面
    # Crop the frame based on the CROP_BOTTOM_PERCENTAGE
    frame = frame[:crop_height, :]

    # --- 車道線偵測流程 ---
    # 應用多層次偵測策略
    combined_mask = preprocess_frame(frame)
    
    # 應用鳥瞰圖轉換到彩色原始影像和二值化影像
    bird_eye_color = cv2.warpPerspective(frame, M, (width, crop_height))
    bird_eye_edges = cv2.warpPerspective(combined_mask, M, (width, crop_height))

    # 在鳥瞰圖上尋找車道線
    lines = cv2.HoughLinesP(bird_eye_edges, 1, np.pi / 180, 60,
                            minLineLength=40, maxLineGap=150)

    # 找到平均車道線，這裡的座標已經是在鳥瞰圖上
    left_line_be, right_line_be = average_slope_intercept(bird_eye_edges, lines)

    # 📉 3. 資料平滑與去雜訊處理
    # ✅ 時間平滑：使用過往帧資料
    # 將當前偵測結果加入歷史隊列
    if left_line_be is not None:
        left_history.append(left_line_be)
    if right_line_be is not None:
        right_history.append(right_line_be)

    # 透過歷史資料計算平均值，使車道線在影片中不會跳動或閃爍
    avg_left_line = np.average(list(left_history), axis=0).astype(int) if left_history else None
    avg_right_line = np.average(list(right_history), axis=0).astype(int) if right_history else None

    # 將二值化鳥瞰圖轉換為三通道，以便在其上繪製彩色車道線
    bird_eye_edges_3ch = cv2.cvtColor(bird_eye_edges, cv2.COLOR_GRAY2BGR)

    # 在彩色的鳥瞰圖上繪製車道線和填充區域
    overlay = draw_lane_area(bird_eye_edges_3ch, avg_left_line, avg_right_line)
    output = cv2.addWeighted(bird_eye_edges_3ch, 1, overlay, 1, 1)

    # 估計轉向並在鳥瞰圖上顯示
    direction = estimate_turn(avg_left_line, avg_right_line)
    cv2.putText(output, direction, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    out.write(output)
    cv2.namedWindow("ADAS Lane Detection - Bird's Eye View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS Lane Detection - Bird's Eye View", 960, 540)
    cv2.imshow("ADAS Lane Detection - Bird's Eye View", output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
