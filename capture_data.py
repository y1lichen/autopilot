import cv2
import mss
import numpy as np
import time
import os
import csv
from pynput import keyboard

# 控制狀態：只管左右
control_state = {"steer": 0}  # -1 左, 0 直, 1 右

def on_press(key):
    try:
        if key.char == "a":
            control_state["steer"] = -1
        elif key.char == "d":
            control_state["steer"] = 1
    except:
        pass

def on_release(key):
    try:
        if key.char in ["a", "d"]:
            control_state["steer"] = 0
    except:
        pass

def main():
    # 建立資料夾
    run_name = f"run_{int(time.time())}"
    base_dir = os.path.join("dataset", run_name)
    os.makedirs(os.path.join(base_dir, "frames"), exist_ok=True)

    csv_file = open(os.path.join(base_dir, "actions.csv"), "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_id", "timestamp", "steer"])

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主螢幕
        frame_id = 0
        while True:
            frame_id += 1
            # 擷取螢幕 (原始大小)
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            # 存圖片（高品質 JPEG）
            filename = f"{frame_id:06d}.jpg"
            cv2.imwrite(os.path.join(base_dir, "frames", filename), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 存動作
            writer.writerow([
                filename,
                time.time(),
                control_state["steer"]
            ])

            # 顯示畫面（可以縮小顯示，不影響存檔）
            show = cv2.resize(frame, (960, 540))  # 只為了顯示不卡
            cv2.imshow("ETS2 Recorder", show)

            if cv2.waitKey(1) == ord("q"):
                break

    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

