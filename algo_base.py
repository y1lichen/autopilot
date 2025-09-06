import cv2
import numpy as np
import os
from collections import deque

# 參數設定
# Parameter Configuration
# ✅ 調整此處的 maxlen 數值以控制反應速度。數值越小，反應越快，但可能抖動會增加。
# 佇列現在儲存的是多項式係數
left_history = deque(maxlen=5) 
right_history = deque(maxlen=5)

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

# 🧠 2. 智能化車道線擬合流程 - 滑動視窗演算法
def find_lane_pixels_and_fit_poly(binary_warped):
    """
    使用滑動視窗演算法在二值化鳥瞰圖上尋找車道線像素，並擬合多項式曲線。
    """
    # 計算直方圖以找到車道線的底部起點
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 滑動視窗設定
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 100
    minpix = 50
    
    left_lane_inds = []
    right_lane_inds = []

    # 遍歷所有滑動視窗
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_x_low_l = leftx_current - margin
        win_x_high_l = leftx_current + margin
        win_x_low_r = rightx_current - margin
        win_x_high_r = rightx_current + margin
        
        # 識別視窗內的非零像素
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_x_low_l) & (nonzerox < win_x_high_l)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_x_low_r) & (nonzerox < win_x_high_r)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # 如果找到足夠多的像素，更新下一個視窗的中心位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # 將所有像素索引串連起來
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass
    
    # 提取車道線像素的座標
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_fit = None
    right_fit = None
    
    # 使用高斯權重進行多項式擬合
    # Fit a second order polynomial to the left and right lane points with Gaussian weighting
    if len(leftx) > 0:
        # 計算左車道線點的平均 x 位置作為中心
        left_mean_x = np.mean(leftx)
        # 設定標準差
        sigma = 30
        # 計算高斯權重
        left_weights = np.exp(-((leftx - left_mean_x)**2) / (2 * sigma**2))
        try:
            left_fit = np.polyfit(lefty, leftx, 2, w=left_weights)
        except TypeError:
            left_fit = None

    if len(rightx) > 0:
        # 計算右車道線點的平均 x 位置作為中心
        right_mean_x = np.mean(rightx)
        # 設定標準差
        sigma = 30
        # 計算高斯權重
        right_weights = np.exp(-((rightx - right_mean_x)**2) / (2 * sigma**2))
        try:
            right_fit = np.polyfit(righty, rightx, 2, w=right_weights)
        except TypeError:
            right_fit = None

    return left_fit, right_fit

def draw_lane_area(image, left_fit, right_fit):
    lane_image = np.zeros_like(image)
    if left_fit is None or right_fit is None:
        return lane_image

    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # 將擬合的多項式點轉換為繪圖用的多邊形
    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((left_line_pts, right_line_pts))

    cv2.fillPoly(lane_image, np.int32(pts), (0, 255, 0))
    cv2.polylines(lane_image, np.int32([left_line_pts]), False, (255, 0, 255), 6)
    cv2.polylines(lane_image, np.int32([right_line_pts]), False, (255, 0, 255), 6)
    return lane_image

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
    frame = frame[:crop_height, :]

    # --- 車道線偵測流程 ---
    # 應用多層次偵測策略
    combined_mask = preprocess_frame(frame)
    
    # 應用鳥瞰圖轉換到彩色原始影像和二值化影像
    bird_eye_color = cv2.warpPerspective(frame, M, (width, crop_height))
    bird_eye_edges = cv2.warpPerspective(combined_mask, M, (width, crop_height))

    # 使用滑動視窗在鳥瞰圖上尋找車道線並擬合多項式
    left_fit_poly, right_fit_poly = find_lane_pixels_and_fit_poly(bird_eye_edges)

    # 📉 3. 資料平滑與去雜訊處理
    # ✅ 時間平滑：使用過往帧資料
    # 將當前偵測結果（多項式係數）加入歷史隊列
    if left_fit_poly is not None:
        left_history.append(left_fit_poly)
    if right_fit_poly is not None:
        right_history.append(right_fit_poly)

    # 透過歷史資料計算平均多項式係數
    avg_left_fit = np.average(list(left_history), axis=0) if left_history else None
    avg_right_fit = np.average(list(right_history), axis=0) if right_history else None

    # 將二值化鳥瞰圖轉換為三通道，以便在其上繪製彩色車道線
    bird_eye_edges_3ch = cv2.cvtColor(bird_eye_edges, cv2.COLOR_GRAY2BGR)

    # 在彩色的鳥瞰圖上繪製車道線和填充區域
    overlay = draw_lane_area(bird_eye_edges_3ch, avg_left_fit, avg_right_fit)
    output = cv2.addWeighted(bird_eye_edges_3ch, 1, overlay, 1, 1)

    # 估計轉向並在鳥瞰圖上顯示
    direction = estimate_turn(avg_left_fit, avg_right_fit)
    cv2.putText(output, direction, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    out.write(output)
    cv2.namedWindow("ADAS Lane Detection - Bird's Eye View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS Lane Detection - Bird's Eye View", 960, 540)
    cv2.imshow("ADAS Lane Detection - Bird's Eye View", output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
