import cv2
import numpy as np
import os


class LaneDetector:
    """
    用於平滑車道線偵測結果的類別。
    它會儲存並平均多幀的車道線數據，以減少抖動。
    """
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.smooth_factor = 0.8  # 平滑因子，值越低，平滑效果越強

    def get_smoothed_lanes(self, new_left_fit, new_right_fit):
        # 對左右車道線的擬合係數進行平滑化
        if self.left_fit is None:
            self.left_fit = new_left_fit
        else:
            if new_left_fit is not None:
                self.left_fit = (self.left_fit * self.smooth_factor) + (new_left_fit * (1 - self.smooth_factor))

        if self.right_fit is None:
            self.right_fit = new_right_fit
        else:
            if new_right_fit is not None:
                self.right_fit = (self.right_fit * self.smooth_factor) + (new_right_fit * (1 - self.smooth_factor))
        
        return self.left_fit, self.right_fit


def preprocess_frame(frame):
    """
    使用自適應二值化和形態學處理來準備影像。
    輸入: 影像 (BGR)
    輸出: 二值化影像 (8-bit 灰階)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊，降低雜訊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自適應二值化 (反轉版本，讓線條白、背景黑)
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        7  
    )

    # 形態學處理
    kernel = np.ones((3, 3), np.uint8)
    
    # 閉運算 (補上斷掉的線條)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 開運算 (去除小的白點)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 再做一次模糊，讓邊緣更平滑
    cleaned = cv2.medianBlur(opened, 5)

    return cleaned

def get_perspective_transforms(frame):
    """
    取得 Bird’s Eye View (透視轉換) 的轉換矩陣。
    輸入: BGR 或灰階 frame
    輸出: 正向和反向的轉換矩陣 M 和 Minv
    """
    height, width = frame.shape[:2]
    src = np.float32([
        (int(0.1 * width), height),
        (int(0.9 * width), height),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.4 * width), int(0.6 * height))
    ])

    dst = np.float32([
        (int(0.2 * width), height),
        (int(0.8 * width), height),
        (int(0.8 * width), 0),
        (int(0.2 * width), 0)
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

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
    minpix = 50
    
    new_left_fit = None
    new_right_fit = None

    if detector.left_fit is not None and detector.right_fit is not None:
        left_lane_inds = ((nonzerox > (detector.left_fit[0]*(nonzeroy**2) + detector.left_fit[1]*nonzeroy + detector.left_fit[2] - margin)) & 
                          (nonzerox < (detector.left_fit[0]*(nonzeroy**2) + detector.left_fit[1]*nonzeroy + detector.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (detector.right_fit[0]*(nonzeroy**2) + detector.right_fit[1]*nonzeroy + detector.right_fit[2] - margin)) & 
                           (nonzerox < (detector.right_fit[0]*(nonzeroy**2) + detector.right_fit[1]*nonzeroy + detector.right_fit[2] + margin)))
    else:
        histogram = np.sum(binary_frame[int(height * 0.75):, :], axis=0)

        x_values = np.arange(width)
        mid_x = width / 2
        initial_weights = 1 - ((x_values - mid_x) / (width/2))**2
        initial_weighted_histogram = histogram * initial_weights

        midpoint = int(initial_weighted_histogram.shape[0] / 2)
        leftx_base_initial = np.argmax(initial_weighted_histogram[:midpoint])
        rightx_base_initial = np.argmax(initial_weighted_histogram[midpoint:]) + midpoint

        left_peak_x = leftx_base_initial
        right_peak_x = rightx_base_initial
        sigma = width * 0.05
        
        left_gaussian_weights = np.exp(-((x_values - left_peak_x)**2) / (2 * sigma**2))
        right_gaussian_weights = np.exp(-((x_values - right_peak_x)**2) / (2 * sigma**2))
        
        final_weights = left_gaussian_weights + right_gaussian_weights
        final_weighted_histogram = histogram * final_weights
        
        leftx_base = np.argmax(final_weighted_histogram[:midpoint])
        rightx_base = np.argmax(final_weighted_histogram[midpoint:]) + midpoint

        lane_width_pixels = rightx_base - leftx_base
        min_lane_width = int(width * 0.3)
        max_lane_width = int(width * 0.6)

        if not (min_lane_width < lane_width_pixels < max_lane_width):
            estimated_leftx = rightx_base - int(width * 0.4) 
            if 0 < estimated_leftx < midpoint:
                leftx_base = estimated_leftx

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

    outlier_threshold = int(width / 10)

    if len(leftx) > minpix:
        initial_left_fit = np.polyfit(lefty, leftx, 2)
        initial_left_fitx = initial_left_fit[0]*lefty**2 + initial_left_fit[1]*lefty + initial_left_fit[2]
        left_dist_from_fit = np.abs(leftx - initial_left_fitx)
        left_inlier_inds = left_dist_from_fit < outlier_threshold
        
        leftx_inliers = leftx[left_inlier_inds]
        lefty_inliers = lefty[left_inlier_inds]

        if len(leftx_inliers) > minpix:
            left_weights = 1 - ((leftx_inliers - width/2) / (width/2))**2
            left_weights[left_weights < 0] = 0
            new_left_fit = np.polyfit(lefty_inliers, leftx_inliers, 2, w=left_weights)

    if len(rightx) > minpix:
        initial_right_fit = np.polyfit(righty, rightx, 2)
        initial_right_fitx = initial_right_fit[0]*righty**2 + initial_right_fit[1]*righty + initial_right_fit[2]
        right_dist_from_fit = np.abs(rightx - initial_right_fitx)
        right_inlier_inds = right_dist_from_fit < outlier_threshold

        rightx_inliers = rightx[right_inlier_inds]
        righty_inliers = righty[right_inlier_inds]
        
        if len(rightx_inliers) > minpix:
            right_weights = 1 - ((rightx_inliers - width/2) / (width/2))**2
            right_weights[right_weights < 0] = 0
            new_right_fit = np.polyfit(righty_inliers, rightx_inliers, 2, w=right_weights)
        
    return new_left_fit, new_right_fit


def draw_lanes_on_images(bird_eye, binary_bird_eye, smoothed_left_fit, smoothed_right_fit):
    """
    在二值化鳥瞰圖上繪製車道線。
    輸入: 二值化影像、平滑化後的擬合係數。
    輸出: 一個包含車道線的彩色二值化影像。
    """
    height, width = bird_eye.shape[:2]
    
    # 將二值化鳥瞰圖轉換為三通道彩色影像
    # 這樣我們才能在上面繪製彩色的車道線
    result_image = cv2.cvtColor(binary_bird_eye, cv2.COLOR_GRAY2BGR)

    if smoothed_left_fit is not None and smoothed_right_fit is not None:
        ploty = np.linspace(0, height - 1, height)
        left_fitx = smoothed_left_fit[0] * ploty**2 + smoothed_left_fit[1] * ploty + smoothed_left_fit[2]
        right_fitx = smoothed_right_fit[0] * ploty**2 + smoothed_right_fit[1] * ploty + smoothed_right_fit[2]
        
        # 創建車道線的多邊形點，這次不閉合
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)

        # 在影像上繪製左側車道線 (綠色)
        cv2.polylines(result_image, [pts_left], False, (0, 255, 0), 20)
        # 在影像上繪製右側車道線 (綠色)
        cv2.polylines(result_image, [pts_right], False, (0, 255, 0), 20)

    return result_image


# ================================
# Main
# ================================

frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")]
)

if not frame_files:
    raise RuntimeError(f"No frames found in {frames_dir}")

first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]
height = int(height * 2 / 3)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 影片輸出高度調整為單一畫面高度
out = cv2.VideoWriter("output_hough_weighted_lanes.mp4", fourcc, 20.0, (width, height))

# 建立車道線偵測器實例
detector = LaneDetector()

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue
    frame = frame[:height,:]

    M, Minv = get_perspective_transforms(frame)
    bird_eye = cv2.warpPerspective(frame, M, (width, height))
    binary_bird_eye = preprocess_frame(bird_eye)
    
    # 找出車道線擬合係數
    new_left_fit, new_right_fit = find_lane_fits(binary_bird_eye, detector)
    
    # 平滑化車道線的擬合係數
    smoothed_left_fit, smoothed_right_fit = detector.get_smoothed_lanes(new_left_fit, new_right_fit)

    # 繪製車道線並取得單一影像
    final_output = draw_lanes_on_images(bird_eye, binary_bird_eye, smoothed_left_fit, smoothed_right_fit)
    
    # 顯示最終的畫面
    cv2.namedWindow("Combined Lane Detection View", cv2.WINDOW_NORMAL)
    # 視窗大小調整為單一畫面高度
    cv2.resizeWindow("Combined Lane Detection View", 960, 540)
    cv2.imshow("Combined Lane Detection View", final_output)
    
    out.write(final_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
