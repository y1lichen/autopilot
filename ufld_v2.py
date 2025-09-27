import os
import cv2
import time
import numpy as np
import onnxruntime as ort

# =====================
# 參數設定
# =====================
frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])

model_path = "/home/chen/Downloads/resources/ufldv2_culane_res18_320x1600.onnx"
inpHeight, inpWidth = 320, 1600

# =====================
# OnnxRuntime Session
# =====================
session_options = ort.SessionOptions()
session_options.log_severity_level = 2
session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])

# =====================
# CULane 設定
# =====================
num_row = 72
num_col = 81
crop_ratio = 0.6

row_anchor = [(0.42 + i * (1.0 - 0.42) / (num_row - 1)) for i in range(num_row)]
col_anchor = [(0.0 + i * (1.0 - 0.0) / (num_col - 1)) for i in range(num_col)]

# =====================
# Common functions
# =====================
def preprocess(img):
    """Resize -> Crop -> Normalize -> NCHW"""
    img_h, img_w = img.shape[:2]
    # 先 resize 到 (1600, 533)
    resized = cv2.resize(img, (inpWidth, int(inpHeight / crop_ratio)))

    # 再裁切，只保留高度 320 (從底部往上)
    start_y = resized.shape[0] - inpHeight
    cropped = resized[start_y:, :, :]  # shape (320,1600,3)

    # normalize
    cropped = cropped.astype(np.float32) / 255.0
    cropped = cropped.transpose(2, 0, 1)  # HWC -> CHW
    cropped = np.expand_dims(cropped, axis=0)  # NCHW
    return cropped, img_h, img_w


def softmax(x):
    """softmax for 1D array"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def argmax_1(arr, dims):
    """模擬 C# Common.argmax_1"""
    arr = arr.reshape(dims)
    return np.argmax(arr, axis=1).reshape(-1)


def sum_valid(valid, num_cls, num_lane, lane_idx):
    """計算 valid 的數量"""
    cnt = 0
    for k in range(num_cls):
        index = k * num_lane + lane_idx
        if valid[index] != 0:
            cnt += 1
    return cnt

# =====================
# 主迴圈
# =====================
for frame_name in frame_files:
    img_path = os.path.join(frames_dir, frame_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"讀取失敗: {img_path}")
        continue

    input_tensor, img_h, img_w = preprocess(image)

    dt1 = time.time()
    outputs = session.run(None, {"input": input_tensor})
    dt2 = time.time()

    loc_row, loc_col, exist_row, exist_col = outputs

    # shape: [1, num_grid, num_cls, num_lane]
    num_grid_row, num_cls_row, num_lane_row = loc_row.shape[1:]
    num_grid_col, num_cls_col, num_lane_col = loc_col.shape[1:]

    # argmax for existence
    valid_row = argmax_1(exist_row, exist_row.shape)
    valid_col = argmax_1(exist_col, exist_col.shape)

    line_list = [[] for _ in range(4)]

    # =====================
    # row-based lane
    # =====================
    for i in [1, 2]:
        if sum_valid(valid_row, num_cls_row, num_lane_row, i) > num_cls_row * 0.5:
            for k in range(num_cls_row):
                index = k * num_lane_row + i
                if valid_row[index] != 0:
                    max_index = np.argmax(loc_row[0, :, k, i])
                    pred_all = []
                    all_inds = []
                    for ind in range(max(0, max_index - 1), min(num_grid_row - 1, max_index + 1) + 1):
                        pred_all.append(loc_row[0, ind, k, i])
                        all_inds.append(ind)
                    pred_soft = softmax(np.array(pred_all))
                    out_temp = np.sum(pred_soft * np.array(all_inds))
                    x = (out_temp + 0.5) / (num_grid_row - 1.0)
                    y = row_anchor[k]
                    line_list[i].append((int(x * img_w), int(y * img_h)))

    # =====================
    # col-based lane
    # =====================
    for i in [0, 1, 2, 3]:
        if sum_valid(valid_col, num_cls_col, num_lane_col, i) > num_cls_col / 4:
            for k in range(num_cls_col):
                index = k * num_lane_col + i
                if valid_col[index] != 0:
                    max_index = np.argmax(loc_col[0, :, k, i])
                    pred_all = []
                    all_inds = []
                    for ind in range(max(0, max_index - 1), min(num_grid_col - 1, max_index + 1) + 1):
                        pred_all.append(loc_col[0, ind, k, i])
                        all_inds.append(ind)
                    pred_soft = softmax(np.array(pred_all))
                    out_temp = np.sum(pred_soft * np.array(all_inds))
                    y = (out_temp + 0.5) / (num_grid_col - 1.0)
                    x = col_anchor[k]
                    line_list[i].append((int(x * img_w), int(y * img_h)))

    # =====================
    # 畫結果
    # =====================
    result_img = image.copy()
    for line in line_list:
        for p in line:
            cv2.circle(result_img, p, 3, (0, 255, 0), -1)

    print(f"{frame_name} 推理耗時: {(dt2 - dt1)*1000:.2f} ms")

    cv2.imshow("result", result_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
