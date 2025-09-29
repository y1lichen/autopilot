import os
import sys
import cv2
import time
import numpy as np
import onnxruntime as ort

# =====================
# 參數設定
# =====================
CROP_CONFIG = {
    "left_cut": 0.00,
    "right_cut": 0.00,
    "top_cut": 0.00,
    "bottom_cut": 0.4
}

# 資料集路徑
frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])

# ONNX 模型路徑
model_path = "/Users/chen/Downloads/resources/ufldv2_culane_res34_320x1600.onnx"
inpHeight, inpWidth = 320, 1600

# =====================
# OnnxRuntime Session
# =====================
session_options = ort.SessionOptions()
session_options.log_severity_level = 2

if sys.platform == "darwin":
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
print("Session providers:", session.get_providers())

# =====================
# UFLDv2/CULane 設定
# =====================
ROW_LANE_INDICES = [1, 2]  # 中央兩條 (Row-based)
COL_LANE_INDICES = [0, 3]  # 外側兩條 (Col-based)

num_row = 72
num_col = 81
crop_ratio = 0.6

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

row_anchor = [(0.42 + i * (1.0 - 0.42) / (num_row - 1)) for i in range(num_row)]
col_anchor = [(0.0 + i * (1.0 - 0.0) / (num_col - 1)) for i in range(num_col)]

# =====================
# Common functions
# =====================
def preprocess(img):
    img_h, img_w = img.shape[:2]
    resized_h = int(inpHeight / crop_ratio)
    resized = cv2.resize(img, (inpWidth, resized_h))
    start_y = resized.shape[0] - inpHeight
    cropped = resized[start_y:, :, :]
    cropped = cropped.astype(np.float32) / 255.0
    cropped = (cropped - MEAN) / STD
    cropped = cropped.transpose(2, 0, 1)
    cropped = np.expand_dims(cropped, axis=0)
    return cropped, img_h, img_w

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sum_valid(valid, num_cls, num_lane, lane_idx):
    cnt = 0
    for k in range(num_cls):
        index = k * num_lane + lane_idx
        if valid.flatten()[index] != 0:
            cnt += 1
    return cnt

# =====================
# 主迴圈
# =====================
for frame_name in frame_files:
    img_path = os.path.join(frames_dir, frame_name)
    image_orig = cv2.imread(img_path)

    if image_orig is None:
        print(f"讀取失敗: {img_path}")
        continue

    H_orig, W_orig = image_orig.shape[:2]

    # 自定義裁切
    x_start = int(W_orig * CROP_CONFIG["left_cut"])
    x_end = int(W_orig * (1.0 - CROP_CONFIG["right_cut"]))
    y_start = int(H_orig * CROP_CONFIG["top_cut"])
    y_end = int(H_orig * (1.0 - CROP_CONFIG["bottom_cut"]))
    image_cropped = image_orig[y_start:y_end, x_start:x_end]

    # 前處理
    input_tensor, H_crop, W_crop = preprocess(image_cropped)

    # 模型推理
    dt1 = time.time()
    outputs = session.run(None, {"input": input_tensor})
    dt2 = time.time()

    loc_row, loc_col, exist_row, exist_col = outputs

    num_grid_row = loc_row.shape[1]
    num_cls_row = loc_row.shape[2]
    num_lane_row = loc_row.shape[3]

    num_grid_col = loc_col.shape[1]
    num_cls_col = loc_col.shape[2]
    num_lane_col = loc_col.shape[3]

    valid_row = np.argmax(exist_row, axis=1)
    valid_col = np.argmax(exist_col, axis=1)

    line_list = [[] for _ in range(4)]

    # Row-based lanes
    for i in ROW_LANE_INDICES:
        if sum_valid(valid_row, num_cls_row, num_lane_row, i) > num_cls_row * 0.5:
            for k in range(num_cls_row):
                if valid_row[0, k, i] != 0:
                    pred_logits = loc_row[0, :, k, i]
                    max_index = np.argmax(pred_logits)
                    all_inds = list(range(max(0, max_index - 1), min(num_grid_row - 1, max_index + 1) + 1))
                    pred_all = [loc_row[0, ind, k, i] for ind in all_inds]
                    pred_soft = softmax(np.array(pred_all))
                    out_temp = np.sum(pred_soft * np.array(all_inds))
                    x_norm = (out_temp + 0.5) / (num_grid_row - 1.0)
                    y_norm = row_anchor[k]
                    x_on_crop = int(x_norm * W_crop)
                    y_on_crop = int(y_norm * H_crop)
                    x_final = x_on_crop + x_start
                    y_final = y_on_crop + y_start
                    line_list[i].append((x_final, y_final))

    # Col-based lanes
    for i in COL_LANE_INDICES:
        if sum_valid(valid_col, num_cls_col, num_lane_col, i) > num_cls_col / 4:
            for k in range(num_cls_col):
                if valid_col[0, k, i] != 0:
                    pred_logits = loc_col[0, :, k, i]
                    max_index = np.argmax(pred_logits)
                    all_inds = list(range(max(0, max_index - 1), min(num_grid_col - 1, max_index + 1) + 1))
                    pred_all = [loc_col[0, ind, k, i] for ind in all_inds]
                    pred_soft = softmax(np.array(pred_all))
                    out_temp = np.sum(pred_soft * np.array(all_inds))
                    y_norm = (out_temp + 0.5) / (num_grid_col - 1.0)
                    x_norm = col_anchor[k]
                    x_on_crop = int(x_norm * W_crop)
                    y_on_crop = int(y_norm * H_crop)
                    x_final = x_on_crop + x_start
                    y_final = y_on_crop + y_start
                    line_list[i].append((x_final, y_final))

    # =====================
    # 畫車道線
    # =====================
    result_img = image_orig.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

    for i in range(len(line_list)):
        line = line_list[i]
        if not line:
            continue
        color = colors[i % len(colors)]
        line.sort(key=lambda p: p[1])
        for j, p in enumerate(line):
            cv2.circle(result_img, p, 5, color, -1)
            if j > 0:
                prev_p = line[j - 1]
                cv2.line(result_img, prev_p, p, color, 2)

    # =====================
    # 車道中心 & 偏移量 (改成中間高度平均)
    # =====================
    car_center = W_orig // 2
    if line_list[1] and line_list[2]:
        left_lane_x = np.mean([p[0] for p in line_list[1]])
        right_lane_x = np.mean([p[0] for p in line_list[2]])

        # 確保左<右
        if left_lane_x > right_lane_x:
            left_lane_x, right_lane_x = right_lane_x, left_lane_x

        lane_center = int((left_lane_x + right_lane_x) / 2)
        offset = lane_center - car_center

        # 自動閾值
        lane_width = right_lane_x - left_lane_x
        threshold = max(int(lane_width * 0.15), 20)  # 至少10px

        if abs(offset) < threshold:
            steer_cmd = "Keep Straight"
        elif offset >= -threshold:
            steer_cmd = "Turn Right"
        elif offset <= threshold:
            steer_cmd = "Keep Left"

        # 畫中心線
        cv2.line(result_img, (car_center, H_orig), (car_center, H_orig - 100), (255, 255, 255), 2)
        cv2.line(result_img, (lane_center, H_orig), (lane_center, H_orig - 100), (0, 255, 255), 2)

        # 顯示文字
        cv2.putText(result_img, f"Offset: {offset:.1f}px", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_img, steer_cmd, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    print(f"{frame_name} 推理耗時: {(dt2 - dt1) * 1000:.2f} ms. 裁切區域: ({x_start},{y_start}) to ({x_end},{y_end})")

    cv2.imshow("result", result_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
