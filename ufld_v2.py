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
    "left_cut": 0.35,
    "right_cut": 0.20,
    "top_cut": 0.20,
    "bottom_cut": 0.40
}
# ------------------------------------------------

frames_dir = "dataset/run_1756133797/frames"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])

model_path = "/Users/chen/Downloads/resources/ufldv2_culane_res18_320x1600.onnx"
# model_path = "/Users/chen/Downloads/resources/ufldv2_curvelanes_res18_800x1600.onnx"
inpHeight, inpWidth = 800, 1600

# =====================
# OnnxRuntime Session
# =====================
session_options = ort.SessionOptions()
session_options.log_severity_level = 2
providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

# =====================
# CULane 設定
# =====================
num_row = 72
num_col = 81
crop_ratio = 0.6

# 這些錨點是根據 UFLDv2 的訓練方式，它們被設計來涵蓋圖像的特定垂直範圍（例如 CULane 覆蓋了 0.42 到 1.0 的垂直比例）。
# 即使您進行了初步裁切，這些錨點仍必須保持不變，因為它們定義了模型輸出的含義。
row_anchor = [(0.42 + i * (1.0 - 0.42) / (num_row - 1)) for i in range(num_row)]
col_anchor = [(0.0 + i * (1.0 - 0.0) / (num_col - 1)) for i in range(num_col)]

# =====================
# Common functions
# =====================
def preprocess(img):
    """
    UFLDv2 專屬前處理: Resize -> CULane Crop (320x1600) -> Normalize -> NCHW
    注意：此處的 img 是經過使用者自定義裁切後的圖片。
    """
    img_h, img_w = img.shape[:2]
    
    # 1. 先將裁切後的圖片 resize 到 (1600, 533)
    # UFLDv2 模型要求輸入尺寸為 320x1600，但因為訓練時使用了 crop_ratio=0.6，
    # 故先 resize 到 1600x(320/0.6) = 1600x533.33
    resized = cv2.resize(img, (inpWidth, int(inpHeight / crop_ratio)))

    # 2. 再裁切，只保留高度 320 (從底部往上)
    # 這是 CULane 的標準輸入，只關注圖像的底部 60% 區域。
    start_y = resized.shape[0] - inpHeight
    cropped = resized[start_y:, :, :]  # shape (320,1600,3)

    # 3. normalize & NCHW
    cropped = cropped.astype(np.float32) / 255.0
    cropped = cropped.transpose(2, 0, 1)  # HWC -> CHW
    cropped = np.expand_dims(cropped, axis=0)  # NCHW
    
    # 返回 (1600, 320) 的輸入張量，以及被裁切圖片的原始尺寸 (img_h, img_w)
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
    # img_h, img_w 這裡會是 image_cropped 的尺寸 (H_crop, W_crop)
    input_tensor, H_crop, W_crop = preprocess(image_cropped)

    # 4. 模型推理
    dt1 = time.time()
    outputs = session.run(None, {"input": input_tensor})
    dt2 = time.time()

    loc_row, loc_col, exist_row, exist_col = outputs

    num_grid_row, num_cls_row, num_lane_row = loc_row.shape[1:]
    num_grid_col, num_cls_col, num_lane_col = loc_col.shape[1:]

    valid_row = argmax_1(exist_row, exist_row.shape)
    valid_col = argmax_1(exist_col, exist_col.shape)

    # line_list 儲存的是相對於 image_cropped 的座標
    line_list = [[] for _ in range(4)]

    # =====================
    # 5. 後處理 & 座標回溯
    # =====================
    
    # row-based lane (i=1, 2 for the two main lanes)
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
                    
                    # 座標 (x_norm, y_norm) 是相對於 UFLDv2 內部處理後的座標
                    x_norm = (out_temp + 0.5) / (num_grid_row - 1.0)
                    y_norm = row_anchor[k] # 錨點 y_norm (0.42~1.0) 是相對於 UFLDv2 輸入圖像的
                    
                    # 將正規化座標轉換為裁切圖片 (image_cropped) 上的像素座標
                    x_on_crop = int(x_norm * W_crop)
                    y_on_crop = int(y_norm * H_crop)

                    # ** 座標回溯：加上自定義裁切的起點偏移量 **
                    x_final = x_on_crop + x_start
                    y_final = y_on_crop + y_start
                    
                    line_list[i].append((x_final, y_final))

    # col-based lane (i=0, 1, 2, 3 for all four lanes)
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

                    # 座標 (x_norm, y_norm) 是相對於 UFLDv2 內部處理後的座標
                    y_norm = (out_temp + 0.5) / (num_grid_col - 1.0)
                    x_norm = col_anchor[k] # 錨點 x_norm (0.0~1.0) 是相對於 UFLDv2 輸入圖像的

                    # 將正規化座標轉換為裁切圖片 (image_cropped) 上的像素座標
                    x_on_crop = int(x_norm * W_crop)
                    y_on_crop = int(y_norm * H_crop)

                    # ** 座標回溯：加上自定義裁切的起點偏移量 **
                    x_final = x_on_crop + x_start
                    y_final = y_on_crop + y_start
                    
                    line_list[i].append((x_final, y_final))

    # =====================
    # 6. 畫結果 (直接在原始圖片上繪製)
    # =====================
    result_img = image_orig.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)] # 不同的顏色來區分車道線
    
    # 只需要 i=1, 2 (row-based) 和 i=0, 3 (col-based)
    for i in range(len(line_list)):
        line = line_list[i]
        color = colors[i % len(colors)]
        
        # 排序：確保點是從上到下或從左到右連接的
        line.sort(key=lambda p: p[1]) 

        for j, p in enumerate(line):
            # 畫出點
            cv2.circle(result_img, p, 5, color, -1)
            
            # 畫出連線 (可選，但能更清晰地顯示車道線)
            if j > 0:
                prev_p = line[j-1]
                cv2.line(result_img, prev_p, p, color, 2)


    print(f"{frame_name} 推理耗時: {(dt2 - dt1)*1000:.2f} ms. 裁切區域: ({x_start},{y_start}) to ({x_end},{y_end})")

    cv2.imshow("result", result_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
