import os
import sys
import cv2
import time
import numpy as np
import onnxruntime as ort

# =====================
# 參數設定 (調整為 TuSimple)
# =====================
CROP_CONFIG = {
    "left_cut": 0.00,
    "right_cut": 0.00,
    "top_cut": 0.00,
    "bottom_cut": 0.4 # 保持原始底部裁切不變，但請注意 TuSimple 原始設定通常是 0.2 (1 - crop_ratio)
}

# 資料集路徑
frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])

# ONNX 模型路徑 (請確認這是 TuSimple 訓練的模型)
model_path = "/home/chen/Downloads/resources/ufldv2_tusimple_res34_320x800.onnx"
inpHeight, inpWidth = 320, 800 # TuSimple 的輸入尺寸

# =====================
# OnnxRuntime Session
# =====================
session_options = ort.SessionOptions()
session_options.log_severity_level = 2

if sys.platform == "darwin":
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
else:
    # 建議使用 OpenVINOExecutionProvider 或 CUDAExecutionProvider
    providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
print("Session providers:", session.get_providers())

# =====================
# UFLDv2/TuSimple 設定
# =====================
ROW_LANE_INDICES = [1, 2]  # 中央兩條 (Row-based)
COL_LANE_INDICES = [0, 3]  # 外側兩條 (Col-based)

# TuSimple/UFLDv2 的錨點設定
num_row = 56 # TuSimple 的行錨點數
num_col = 100 # TuSimple 的網格數
crop_ratio = 0.8 # TuSimple 的裁切比例

# TuSimple 的標準錨點計算 (基於 720x1280 影像)
# Row anchors: np.linspace(160, 710, 56) / 720
row_anchor = np.linspace(160, 710, num_row) / 720.0
# Col anchors: np.linspace(0, 1, 100)
col_anchor = np.linspace(0.0, 1.0, num_col)

# 重新計算 row_anchor, 使其與 inpHeight 匹配
# 原始的 row_anchor 是基於原始圖片的 y 座標比例
# 由於 preprocess 已經處理了裁切，這裡使用模型預期的比例
# 但因為原始的 row_anchor 是基於 720, 我們先將其轉換為比例
# 模型的輸入是從 1 - crop_ratio (0.2) 開始到 1.0 的區域
# 這裡使用一個簡化的比例，但更精確的應該是使用模型訓練時的比例

# 重新定義 row_anchor 保持一致性 (如果模型是基於標準 TuSimple 訓練)
# 因為 UFLDv2 的 row_anchor 是針對原始圖片 y 座標的比例，所以直接使用比例即可
row_anchor = [(160 + i * (710 - 160) / (num_row - 1)) / 720.0 for i in range(num_row)]
col_anchor = [(0.0 + i * (1.0 - 0.0) / (num_col - 1)) for i in range(num_col)]


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

car_center_offset = 30
# =====================
# Common functions
# =====================
def preprocess(img):
    img_h, img_w = img.shape[:2]
    
    # 根據 crop_ratio 調整縮放高度
    resized_h = int(inpHeight / crop_ratio) # 320 / 0.8 = 400
    resized = cv2.resize(img, (inpWidth, resized_h))
    
    # 裁切，取底部 inpHeight (320) 高度的部分
    start_y = resized.shape[0] - inpHeight # 400 - 320 = 80
    cropped = resized[start_y:, :, :]
    
    # 歸一化和標準化
    cropped = cropped.astype(np.float32) / 255.0
    cropped = (cropped - MEAN) / STD
    
    # 改變維度順序 (HWC -> CHW) 並添加 batch 維度
    cropped = cropped.transpose(2, 0, 1)
    cropped = np.expand_dims(cropped, axis=0)
    return cropped, img_h, img_w

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sum_valid(valid, num_cls, num_lane, lane_idx):
    cnt = 0
    # valid 的 shape 是 (1, num_cls, num_lane)
    for k in range(num_cls):
        if valid[0, k, lane_idx] != 0:
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

    # 自定義裁切 (TuSimple 通常是 0.8 裁切，即底部 80%)
    # 但這裡保留您的自定義裁切 CROP_CONFIG
    x_start = int(W_orig * CROP_CONFIG["left_cut"])
    x_end = int(W_orig * (1.0 - CROP_CONFIG["right_cut"]))
    y_start = int(H_orig * CROP_CONFIG["top_cut"])
    y_end = int(H_orig * (1.0 - CROP_CONFIG["bottom_cut"]))
    image_cropped = image_orig[y_start:y_end, x_start:x_end]

    # 前處理 (這裡的 H_crop, W_crop 應該是 image_cropped 的 H/W)
    input_tensor, H_crop, W_crop = preprocess(image_cropped)
    H_model, W_model = inpHeight, inpWidth

    # 模型推理
    dt1 = time.time()
    outputs = session.run(None, {"input": input_tensor})
    dt2 = time.time()

    loc_row, loc_col, exist_row, exist_col = outputs

    num_grid_row = loc_row.shape[1]
    num_cls_row = loc_row.shape[2]
    num_lane_row = loc_row.shape[3] # 4

    num_grid_col = loc_col.shape[1]
    num_cls_col = loc_col.shape[2]
    num_lane_col = loc_col.shape[3] # 4

    # 獲取是否存在
    valid_row = np.argmax(exist_row, axis=1) # (1, num_cls_row, num_lane_row)
    valid_col = np.argmax(exist_col, axis=1) # (1, num_cls_col, num_lane_col)

    line_list = [[] for _ in range(4)]

    # Row-based lanes (TuSimple 主要使用 Row-based)
    for i in ROW_LANE_INDICES:
        # 檢查車道 i 是否存在 (超過一半的行錨點有預測)
        if sum_valid(valid_row, num_cls_row, num_lane_row, i) > num_cls_row * 0.5:
            for k in range(num_cls_row):
                if valid_row[0, k, i] != 0:
                    pred_logits = loc_row[0, :, k, i]
                    max_index = np.argmax(pred_logits)
                    
                    # 軟化預測 (Softmax Aggregation)
                    all_inds = list(range(max(0, max_index - 1), min(num_grid_row - 1, max_index + 1) + 1))
                    pred_all = [loc_row[0, ind, k, i] for ind in all_inds]
                    pred_soft = softmax(np.array(pred_all))
                    out_temp = np.sum(pred_soft * np.array(all_inds))
                    
                    # 將網格座標轉換為圖像座標 (X, Y)
                    x_norm = (out_temp + 0.5) / (num_grid_row - 1.0) # 歸一化 X
                    y_norm = row_anchor[k] # 歸一化 Y (基於原始圖片)
                    
                    # 將歸一化座標映射回裁切後的圖片尺寸
                    x_on_crop = int(x_norm * W_crop)
                    
                    # 由於 row_anchor 是基於原始圖片的 y 比例，我們需要考慮裁切的比例。
                    # 簡單來說，如果原始圖片是 H_orig，裁切後是 H_crop，y_norm 是原始比例。
                    # y_on_crop = y_norm * H_orig - y_start
                    y_on_orig = int(y_norm * H_orig)
                    y_on_crop = y_on_orig - y_start # 裁切圖上的 Y
                    
                    # 檢查 y_on_crop 是否在範圍內
                    if 0 <= y_on_crop < H_crop:
                        # 映射回原始圖片座標
                        x_final = x_on_crop + x_start
                        y_final = y_on_crop + y_start
                        line_list[i].append((x_final, y_final))

    # Col-based lanes (TuSimple 也有 Col-based，但通常不如 Row-based 穩定，這裡保留處理)
    for i in COL_LANE_INDICES:
        # 檢查車道 i 是否存在
        if sum_valid(valid_col, num_cls_col, num_lane_col, i) > num_cls_col / 4:
            for k in range(num_cls_col):
                if valid_col[0, k, i] != 0:
                    pred_logits = loc_col[0, :, k, i]
                    max_index = np.argmax(pred_logits)
                    
                    # 軟化預測 (Softmax Aggregation)
                    all_inds = list(range(max(0, max_index - 1), min(num_grid_col - 1, max_index + 1) + 1))
                    pred_all = [loc_col[0, ind, k, i] for ind in all_inds]
                    pred_soft = softmax(np.array(pred_all))
                    out_temp = np.sum(pred_soft * np.array(all_inds))
                    
                    # 將網格座標轉換為圖像座標 (X, Y)
                    y_norm = (out_temp + 0.5) / (num_grid_col - 1.0) # 歸一化 Y
                    x_norm = col_anchor[k] # 歸一化 X (基於原始圖片)
                    
                    # 將歸一化座標映射回裁切後的圖片尺寸
                    x_on_crop = int(x_norm * W_crop) # 裁切圖上的 X
                    
                    # 由於 col_anchor 是基於原始圖片的 X 比例，y_norm 是針對模型輸入的 y 比例 (320x800)
                    # y_on_crop 已經是針對裁切圖 (H_crop, W_crop) 的 Y 座標
                    y_on_crop = int(y_norm * H_crop)

                    # 檢查 y_on_crop 是否在範圍內
                    if 0 <= y_on_crop < H_crop:
                        # 映射回原始圖片座標
                        x_final = x_on_crop + x_start
                        y_final = y_on_crop + y_start
                        line_list[i].append((x_final, y_final))

    # =====================
    # 畫車道線
    # =====================
    result_img = image_orig.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)] # BGR

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
    # 車道中心 & 偏移量
    # =====================
    car_center = W_orig // 2
    car_center = car_center + car_center_offset
    steer_cmd = "No Lane Detected"
    offset = 0.0

    # 僅使用自車道線 (index 1 和 2) 來計算中心
    if line_list[1] and line_list[2]:
        # 計算兩條車道線 X 座標的平均值
        left_lane_x = np.mean([p[0] for p in line_list[1]])
        right_lane_x = np.mean([p[0] for p in line_list[2]])

        # 確保左 < 右
        if left_lane_x > right_lane_x:
            left_lane_x, right_lane_x = right_lane_x, left_lane_x

        lane_center = int((left_lane_x + right_lane_x) / 2)
        offset = lane_center - car_center

        # 自動閾值
        lane_width = right_lane_x - left_lane_x
        threshold = max(int(lane_width * 0.15), 20)  # 至少 20px

        if abs(offset) < threshold:
            steer_cmd = "Keep Straight"
        elif offset > 0: # offset > threshold
            steer_cmd = "Turn Right"
        elif offset < 0: # offset < -threshold
            steer_cmd = "Keep Left"

        # 畫中心線
        cv2.line(result_img, (car_center, H_orig), (car_center, H_orig - 100), (255, 255, 255), 2) # 車輛中心 (白色)
        cv2.line(result_img, (lane_center, H_orig), (lane_center, H_orig - 100), (0, 255, 255), 2) # 車道中心 (黃色)

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