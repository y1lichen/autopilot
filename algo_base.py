import cv2
import numpy as np
import os
from collections import deque

class LaneDetector:
    """
    用於平滑車道線偵測結果的類別。
    它會儲存並平均多幀的車道線數據，以減少抖動。
    """
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.smooth_factor = 0.5 # 平滑化因子，值越小越平滑

    def get_smoothed_lanes(self, new_left_fit, new_right_fit):
        """
        對左右車道線的擬合係數進行平滑化處理。
        """
        # 如果是第一次偵測，直接賦值
        if self.left_fit is None and new_left_fit is not None:
            self.left_fit = new_left_fit
        # 否則，進行加權平均
        elif new_left_fit is not None:
            self.left_fit = (self.left_fit * self.smooth_factor) + (new_left_fit * (1 - self.smooth_factor))

        if self.right_fit is None and new_right_fit is not None:
            self.right_fit = new_right_fit
        elif new_right_fit is not None:
            self.right_fit = (self.right_fit * self.smooth_factor) + (new_right_fit * (1 - self.smooth_factor))
        
        return self.left_fit, self.right_fit


# 參數化 Bird's Eye View 的來源點，方便手動調校
# These points are relative to the image dimensions (width, height).
# These are the corners of the trapezoid ROI in the original image.
BEV_SRC_POINTS_REL = np.float32([
    [0.4, 0.95],
    [0.65, 0.95],
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

# 🧠 2. 智能化車道線擬合流程
def find_lane_fits(binary_frame, detector):
    """
    使用多項式擬合找到車道線的擬合係數。
    輸入: 預處理過的二值化影像，LaneDetector 實例
    輸出: 左右車道線的擬合係數
    """
    height, width = binary_frame.shape[:2]
    
    nonzero = binary_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    nwindows = 9
    window_height = int(height / nwindows)
    margin = 50
    minpix = 50  # 降低 minpix 值以更好地偵測虛線
    
    new_left_fit = None
    new_right_fit = None

    if detector.left_fit is not None and detector.right_fit is not None:
        # 如果有歷史資料，只在先前偵測的車道線附近尋找，反應會更快
        left_lane_inds = ((nonzerox > (detector.left_fit[0]*(nonzeroy**2) + detector.left_fit[1]*nonzeroy + detector.left_fit[2] - margin)) & 
                          (nonzerox < (detector.left_fit[0]*(nonzeroy**2) + detector.left_fit[1]*nonzeroy + detector.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (detector.right_fit[0]*(nonzeroy**2) + detector.right_fit[1]*nonzeroy + detector.right_fit[2] - margin)) & 
                           (nonzerox < (detector.right_fit[0]*(nonzeroy**2) + detector.right_fit[1]*nonzeroy + detector.right_fit[2] + margin)))
    else:
        # 第一次偵測或歷史資料遺失時，使用直方圖分析尋找起點
        histogram = np.sum(binary_frame[int(height * 0.75):, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        left_lane_inds = []
        right_lane_inds = []
        leftx_current = leftx_base
        rightx_current = rightx_base

        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_x_left_low = leftx_current - margin
            win_x_left_high = leftx_current + margin
            win_x_right_low = rightx_current - margin
            win_x_right_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_x_left_low) & (nonzerox < win_x_left_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_x_right_low) & (nonzerox < win_x_right_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # 設定高斯分佈的參數
    # The standard deviation controls how fast the weight drops off.
    # A smaller value means a narrower bell curve (more focus on the center).
    sigma = width / 6
    center_x = width / 2
    
    if len(leftx) > minpix:
        # 計算左側車道線點的高斯權重
        left_weights = np.exp(-((leftx - center_x)**2) / (2 * sigma**2))
        new_left_fit = np.polyfit(lefty, leftx, 2, w=left_weights)

    if len(rightx) > minpix:
        # 計算右側車道線點的高斯權重
        right_weights = np.exp(-((rightx - center_x)**2) / (2 * sigma**2))
        new_right_fit = np.polyfit(righty, rightx, 2, w=right_weights)
    
    return new_left_fit, new_right_fit


def draw_lanes_on_images(base_image, smoothed_left_fit, smoothed_right_fit):
    """
    在基礎影像上繪製車道線和區域。
    輸入: 基礎影像 (彩色或三通道二值化)、平滑化後的擬合係數。
    輸出: 包含車道線和區域的影像。
    """
    height, width = base_image.shape[:2]
    
    # 創建一個空白影像用於繪製
    lane_image = np.zeros_like(base_image)

    if smoothed_left_fit is not None and smoothed_right_fit is not None:
        ploty = np.linspace(0, height - 1, height)
        left_fitx = smoothed_left_fit[0] * ploty**2 + smoothed_left_fit[1] * ploty + smoothed_left_fit[2]
        right_fitx = smoothed_right_fit[0] * ploty**2 + smoothed_right_fit[1] * ploty + smoothed_right_fit[2]
        
        # 創建車道線的多邊形點
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)

        # 繪製車道線區域
        pts = np.hstack((pts_left, pts_right[:, ::-1]))
        cv2.fillPoly(lane_image, pts, (0, 255, 0))

        # 在影像上繪製車道線 (綠色)
        cv2.polylines(lane_image, [pts_left], False, (255, 0, 255), 20)
        cv2.polylines(lane_image, [pts_right], False, (255, 0, 255), 20)
    
    # 將繪製的車道線區域與基礎影像合併
    result = cv2.addWeighted(base_image, 1, lane_image, 0.5, 0)
    return result

def estimate_turn(left, right):
    if left is None or right is None:
        return "Detecting..."
    # 根據多項式係數估計曲率
    # 這是基於多項式二階導數的曲率計算
    left_curverad = ((1 + (2 * left[0] * 720 + left[1])**2)**1.5) / np.absolute(2 * left[0])
    right_curverad = ((1 + (2 * right[0] * 720 + right[1])**2)**1.5) / np.absolute(2 * right[0])
    avg_curverad = (left_curverad + right_curverad) / 2
    
    # 判斷轉向
    if avg_curverad > 2000:
        return "Straight"
    elif left[0] < 0:
        return "Turning Right"
    else:
        return "Turning Left"


# ================================
# Main
# ================================
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

# 建立車道線偵測器實例
detector = LaneDetector()
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

    # 使用多項式擬合來尋找車道線
    new_left_fit, new_right_fit = find_lane_fits(bird_eye_edges, detector)
    
    # 平滑化車道線的擬合係數
    smoothed_left_fit, smoothed_right_fit = detector.get_smoothed_lanes(new_left_fit, new_right_fit)

    # 將二值化鳥瞰圖轉換為三通道，以便在其上繪製彩色車道線
    bird_eye_edges_3ch = cv2.cvtColor(bird_eye_edges, cv2.COLOR_GRAY2BGR)

    # 在三通道二值化圖上繪製車道線和填充區域
    final_output = draw_lanes_on_images(bird_eye_edges_3ch, smoothed_left_fit, smoothed_right_fit)

    # 估計轉向並在鳥瞰圖上顯示
    direction = estimate_turn(smoothed_left_fit, smoothed_right_fit)
    cv2.putText(final_output, direction, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    out.write(final_output)
    cv2.namedWindow("ADAS Lane Detection - Bird's Eye View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS Lane Detection - Bird's Eye View", 960, 540)
    cv2.imshow("ADAS Lane Detection - Bird's Eye View", final_output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
