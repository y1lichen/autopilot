import os
import cv2
import time
import numpy as np
import onnxruntime as ort

# =====================
# 參數設定
# =====================
# --- 自定義裁切比例設定 (作為原始圖片的比例) ---
# 左 0.35, 右 0.2, 上 0.2, 下 0.4
CROP_CONFIG = {
    "left_cut": 0.00,
    "right_cut": 0.00,
    "top_cut": 0.00,
    "bottom_cut": 0.4
}
# ------------------------------------------------

# 替換成您的資料集路徑
frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])

# 替換成您的 ONNX 模型路徑
model_path = "/Users/chen/Downloads/resources/ufldv2_culane_res34_320x1600.onnx"
inpHeight, inpWidth = 320, 1600

# =====================
# OnnxRuntime Session
# =====================
session_options = ort.SessionOptions()
session_options.log_severity_level = 2
providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

# =====================
# UFLDv2/CULane 設定
# =====================
# 預期的車道線索引：
# 0: 左側車道線 (Col-based)
# 1: 左主要車道線 (Row-based)
# 2: 右主要車道線 (Row-based)
# 3: 右側車道線 (Col-based)
ROW_LANE_INDICES = [1, 2] # 使用 Row-based 預測中央的兩條車道線 (1, 2)
COL_LANE_INDICES = [0, 3] # 使用 Col-based 預測外側的兩條車道線 (0, 3)

num_row = 72
num_col = 81
crop_ratio = 0.6 # CULane 的標準裁切比例 (只看底部 60% 區域)

# UFLDv2/ImageNet 標準的 Mean/Std 正規化參數
# 必須與模型訓練時使用的參數一致
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# 這些錨點必須保持不變，因為它們定義了模型輸出的含義。
# 垂直錨點 (Y-axis ratio on the 533-height image)
row_anchor = [(0.42 + i * (1.0 - 0.42) / (num_row - 1)) for i in range(num_row)]
# 水平錨點 (X-axis ratio on the 1600-width image)
col_anchor = [(0.0 + i * (1.0 - 0.0) / (num_col - 1)) for i in range(num_col)]


# =====================
# Common functions
# =====================
def preprocess(img):
    """
    UFLDv2 專屬前處理: Resize -> CULane Crop (320x1600) -> Normalize -> NCHW
    注意：此處的 img 是經過使用者自定義裁切後的圖片 (image_cropped)。
    """
    img_h, img_w = img.shape[:2]
    
    # 1. 先將裁切後的圖片 resize 到 (1600, 533)
    # UFLDv2 模型要求輸入尺寸為 320x1600，但因為訓練時使用了 crop_ratio=0.6，
    # 故先 resize 到 1600x(320/0.6) = 1600x533.33
    resized_h = int(inpHeight / crop_ratio)
    resized = cv2.resize(img, (inpWidth, resized_h))

    # 2. 再裁切，只保留高度 320 (從底部往上)
    # 這是 CULane 的標準輸入，只關注圖像的底部 60% 區域。
    start_y = resized.shape[0] - inpHeight
    cropped = resized[start_y:, :, :]  # shape (320,1600,3)

    # 3. 正規化 (關鍵修正點) & NCHW
    cropped = cropped.astype(np.float32) / 255.0
    # 應用 ImageNet 的 Mean/Std 正規化
    cropped = (cropped - MEAN) / STD
    
    cropped = cropped.transpose(2, 0, 1)  # HWC -> CHW
    cropped = np.expand_dims(cropped, axis=0)  # NCHW
    
    # 返回 (1600, 320) 的輸入張量，以及被裁切圖片的原始尺寸 (img_h, img_w)
    return cropped, img_h, img_w


def softmax(x):
    """softmax for 1D array"""
    # 避免數值溢出 (Overflow)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sum_valid(valid, num_cls, num_lane, lane_idx):
    """計算 valid 的數量"""
    # valid 是一維陣列，需要手動計算該車道線索引的有效點數
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
    
    # 1. 讀取原始圖片
    image_orig = cv2.imread(img_path)

    if image_orig is None:
        print(f"讀取失敗: {img_path}")
        continue

    H_orig, W_orig = image_orig.shape[:2]

    # 2. 執行自定義裁切
    # 計算裁切的起點和終點座標 (在原始圖片上)
    x_start = int(W_orig * CROP_CONFIG["left_cut"])
    x_end = int(W_orig * (1.0 - CROP_CONFIG["right_cut"]))
    y_start = int(H_orig * CROP_CONFIG["top_cut"])
    y_end = int(H_orig * (1.0 - CROP_CONFIG["bottom_cut"]))

    # 進行裁切
    image_cropped = image_orig[y_start:y_end, x_start:x_end]

    # 3. UFLDv2 前處理 (對裁切後的圖片進行)
    # H_crop, W_crop 是 image_cropped 的尺寸
    input_tensor, H_crop, W_crop = preprocess(image_cropped)

    # 4. 模型推理
    dt1 = time.time()
    # loc_row, loc_col, exist_row, exist_col
    outputs = session.run(None, {"input": input_tensor})
    dt2 = time.time()

    loc_row, loc_col, exist_row, exist_col = outputs

    num_grid_row = loc_row.shape[1] # 200
    num_cls_row = loc_row.shape[2] # 72
    num_lane_row = loc_row.shape[3] # 4
    
    num_grid_col = loc_col.shape[1] # 81
    num_cls_col = loc_col.shape[2] # 72
    num_lane_col = loc_col.shape[3] # 4

    # max_indices for existence
    valid_row = np.argmax(exist_row, axis=1) # shape (1, num_cls_row, num_lane_row) -> (num_cls_row, num_lane_row)
    valid_col = np.argmax(exist_col, axis=1)

    # line_list 儲存的是相對於 image_orig 的全域座標
    line_list = [[] for _ in range(4)]

    # =====================
    # 5. 後處理 & 座標回溯
    # =====================
    
    # Row-based lane (主要車道線, 索引 1 和 2)
    for i in ROW_LANE_INDICES:
        # 檢查車道線是否存在 (有效點的數量超過 num_cls_row * 0.5)
        if sum_valid(valid_row, num_cls_row, num_lane_row, i) > num_cls_row * 0.5:
            for k in range(num_cls_row):
                # valid_row 的索引是 k * num_lane + i，但 valid_row.shape 是 (1, num_cls, num_lane)
                # 這裡使用 reshape 且假設 batch size 為 1
                if valid_row[0, k, i] != 0:
                    
                    # Softmax 處理
                    # loc_row shape: (1, num_grid_row, num_cls_row, num_lane_row)
                    pred_logits = loc_row[0, :, k, i]
                    max_index = np.argmax(pred_logits)
                    
                    # 考慮最大值周圍 3 個點進行 Softmax
                    all_inds = list(range(max(0, max_index - 1), min(num_grid_row - 1, max_index + 1) + 1))
                    pred_all = [loc_row[0, ind, k, i] for ind in all_inds]

                    pred_soft = softmax(np.array(pred_all))
                    out_temp = np.sum(pred_soft * np.array(all_inds))
                    
                    # 座標 (x_norm, y_norm) 是相對於 UFLDv2 內部處理後的座標
                    x_norm = (out_temp + 0.5) / (num_grid_row - 1.0) # X 軸比例 (相對於 1600)
                    y_norm = row_anchor[k] # Y 軸比例 (相對於 533)
                    
                    # 將正規化座標轉換為裁切圖片 (image_cropped) 上的像素座標
                    # X: 比例 * W_crop
                    x_on_crop = int(x_norm * W_crop)
                    # Y: 比例 * H_crop
                    y_on_crop = int(y_norm * H_crop)

                    # ** 座標回溯：加上自定義裁切的起點偏移量 **
                    x_final = x_on_crop + x_start
                    y_final = y_on_crop + y_start
                    
                    line_list[i].append((x_final, y_final))

    # Col-based lane (外側車道線, 索引 0 和 3)
    # 注意：這裡只處理 0 和 3，避免與 Row-based 處理的 1 和 2 重複
    for i in COL_LANE_INDICES:
        # 檢查車道線是否存在 (有效點的數量超過 num_cls_col * 0.25)
        if sum_valid(valid_col, num_cls_col, num_lane_col, i) > num_cls_col / 4:
            for k in range(num_cls_col):
                if valid_col[0, k, i] != 0:
                    
                    # Softmax 處理
                    # loc_col shape: (1, num_grid_col, num_cls_col, num_lane_col)
                    pred_logits = loc_col[0, :, k, i]
                    max_index = np.argmax(pred_logits)
                    
                    # 考慮最大值周圍 3 個點進行 Softmax
                    all_inds = list(range(max(0, max_index - 1), min(num_grid_col - 1, max_index + 1) + 1))
                    pred_all = [loc_col[0, ind, k, i] for ind in all_inds]
                    
                    pred_soft = softmax(np.array(pred_all))
                    out_temp = np.sum(pred_soft * np.array(all_inds))

                    # 座標 (x_norm, y_norm) 是相對於 UFLDv2 內部處理後的座標
                    y_norm = (out_temp + 0.5) / (num_grid_col - 1.0) # Y 軸比例 (相對於 533)
                    x_norm = col_anchor[k] # X 軸比例 (相對於 1600)

                    # 將正規化座標轉換為裁切圖片 (image_cropped) 上的像素座標
                    # X: 比例 * W_crop
                    x_on_crop = int(x_norm * W_crop)
                    # Y: 比例 * H_crop
                    y_on_crop = int(y_norm * H_crop)

                    # ** 座標回溯：加上自定義裁切的起點偏移量 **
                    x_final = x_on_crop + x_start
                    y_final = y_on_crop + y_start
                    
                    line_list[i].append((x_final, y_final))

    # =====================
    # 6. 畫結果 (直接在原始圖片上繪製)
    # =====================
    result_img = image_orig.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)] # BGR 顏色來區分車道線 (紅, 綠, 藍, 黃)
    
    # 畫出所有偵測到的車道線
    for i in range(len(line_list)):
        line = line_list[i]
        
        # 排除未偵測到的車道線
        if not line:
            continue
            
        color = colors[i % len(colors)]
        
        # 排序：確保點是從上到下連接的，以便繪製連續的線段
        line.sort(key=lambda p: p[1]) 

        for j, p in enumerate(line):
            # 畫出點
            cv2.circle(result_img, p, 5, color, -1)
            
            # 畫出連線
            if j > 0:
                prev_p = line[j-1]
                # 確保連接的點在水平方向上不要有太大的跳躍，以避免畫出錯誤的線
                # 這裡省略，保持簡單連線
                cv2.line(result_img, prev_p, p, color, 2)


    print(f"{frame_name} 推理耗時: {(dt2 - dt1)*1000:.2f} ms. 裁切區域: ({x_start},{y_start}) to ({x_end},{y_end})")

    cv2.imshow("result", result_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
