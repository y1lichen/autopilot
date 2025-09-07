import cv2
import numpy as np
import os

def preprocess_frame(frame):
    """
    灰階 -> 高斯模糊 -> 邊緣偵測
    輸入: BGR frame
    輸出: edges (單通道灰階影像)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    v = np.median(blur)
    edges = cv2.Canny(blur, int(0.66*v), int(1.33*v))
    return edges

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

    src = np.float32([
        [int(0.4*width), int(0.6*height)],  # 左上
        [int(0.6*width), int(0.6*height)],  # 右上
        [int(0.1*width), height],           # 左下
        [int(0.9*width), height]            # 右下
    ])

    dst = np.float32([
        [int(0.2*width), 0],
        [int(0.8*width), 0],
        [int(0.2*width), height],
        [int(0.8*width), height]
    ])

    # src = np.float32([
    #     [int(0.4*width), int(0.5*height)],   # 左上
    #     [int(0.6*width), int(0.5*height)],   # 右上
    #     [int(0.9*width), height],            # 右下
    #     [int(0.1*width), height]             # 左下
    # ])
    # dst = np.float32([
    #     [int(0.2*width), 0],                 # 左上
    #     [int(0.8*width), 0],                 # 右上
    #     [int(0.8*width), height],            # 右下
    #     [int(0.2*width), height]             # 左下
    # ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

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
    roi = region_of_interest(frame)

    # Step 2: Bird’s Eye View
    bird_eye = warp_perspective(roi)

    # Step 3: Preprocess
    edges = preprocess_frame(bird_eye)

    # 因為 VideoWriter 需要 3 channel，把灰階轉回 BGR
    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 顯示並儲存
    cv2.imshow("ADAS Lane Detection with Bird Eye", output)
    out.write(output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()