import cv2
import numpy as np
import os


def preprocess_frame(frame):
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊，降低雜訊
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # 自適應二值化 (反轉版本，讓線條白、背景黑)
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 高斯比mean更穩定
        cv2.THRESH_BINARY_INV,           # INV => 線條白、背景黑
        15,  # blockSize (奇數, 越大越平滑)
        7    # C (調高可壓掉背景)
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

# def preprocess_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 高斯模糊去掉噪聲
#     blur = cv2.GaussianBlur(gray, (5,5), 0)

#     # 二值化 (只保留亮的)
#     _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

#     # 形態學操作 -> 把斷斷續續的線連起來
#     kernel = np.ones((5,5), np.uint8)
#     mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#     return mask

def region_of_interest(frame):
    """
    在原始 BGR 影像上套 ROI (梯形)
    輸出: 套用遮罩後的 BGR frame
    """
    height, width = frame.shape[:2]

    polygons = np.array([[
        (int(0.1*width), height),        # 左下
        (int(0.9*width), height),        # 右下
        (int(0.6*width), int(0.5*height)),  # 右上
        (int(0.4*width), int(0.5*height))   # 左上
    ]], np.int32)

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked = cv2.bitwise_and(frame, mask)
    return masked

def warp_perspective(frame):
    """
    Bird’s Eye View (透視轉換)
    輸入: BGR 或灰階 frame
    輸出: 俯瞰圖
    """
    height, width = frame.shape[:2]

    # src = np.float32([
    #     [int(0.4*width), int(0.5*height)],  # 左上
    #     [int(0.1*width), height],           # 左下
    #     [int(0.6*width), int(0.5*height)],  # 右上
    #     [int(0.9*width), height]            # 右下
    # ])

    # dst = np.float32([
    #     [0, 0],              # 左上
    #     [0, height],         # 左下
    #     [width, 0],          # 右上
    #     [width, height],     # 右下
    # ])

    src = np.float32([
        (int(0.1*width), height),           # 左下
        (int(0.9*width), height),           # 右下
        (int(0.6*width), int(0.6*height)),  # 右上
        (int(0.4*width), int(0.6*height))   # 左上
    ])

    dst = np.float32([
        (int(0.2*width), height),           # 左下
        (int(0.8*width), height),           # 右下
        (int(0.8*width), 0),                # 右上
        (int(0.2*width), 0)                 # 左上
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

def color_threshold(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    lower_yellow = np.array([15, 50, 80])
    upper_yellow = np.array([35, 255, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return combined_mask
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
out = cv2.VideoWriter("output_bird_eye.mp4", fourcc, 20.0, (width, height))

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue
    frame = frame[:height,:]

    # Step 1: ROI
    # roi = region_of_interest(frame)

    # Step 1: Bird’s Eye View
    bird_eye = warp_perspective(frame)

    # Step 2: 顏色篩選 (白色 + 黃色)
    thresholded = color_threshold(bird_eye)

    # Step 3: Preprocess (Canny)
    edges = preprocess_frame(thresholded)


    # 因為 VideoWriter 需要 3 channel，把灰階轉回 BGR
    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 顯示並儲存
    cv2.imshow("ADAS Lane Detection with Bird Eye", output)
    out.write(output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
