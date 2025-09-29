import cv2
import os
from ultralytics import YOLO

# === 載入你訓練好的 YOLO 模型 ===
model = YOLO("yolo_weight/YOLOv7-tiny.pt")  # 換成你的模型路徑

# === 資料來源 ===
frames_dir = "dataset/run_1755702281/frames"   # 白天
frames_dir = "dataset/run_1755702912/frames"   # 夜晚
frames_dir = "dataset/run_1756133797/frames"   # 其他測試
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])
if not frame_files:
    raise RuntimeError("No frames found!")

# 讀第一張來決定輸出大小
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_lane_yolo.mp4", fourcc, 20.0, (width, height))

for fname in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, fname))
    if frame is None:
        continue

    # === YOLO 推論 (Segmentation) ===
    results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)

    # YOLO 已內建可繪製標註結果
    frame_with_lane = results[0].plot()  

    # 如果你要加方向判斷，可以從 mask 或 bbox 做計算
    # 比方說計算 lane segmentation 的中心線
    # turn_status = "LEFT" / "RIGHT" / "STRAIGHT"
    # cv2.putText(frame_with_lane, turn_status, (30, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.namedWindow("YOLO Lane Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Lane Detection", 960, 540)
    cv2.imshow("YOLO Lane Detection", frame_with_lane)
    out.write(frame_with_lane)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
