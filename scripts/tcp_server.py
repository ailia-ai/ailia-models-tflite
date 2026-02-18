#---------------------------
# ailia MODELS TFLITE TCP SERVER (カメラ画像を3クライアントに送信)
#---------------------------

import os, sys
import argparse
import threading
import time
import cv2
import numpy as np

#---------------------------
# find utils path
#---------------------------

def find_and_append_util_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        potential_util_path = os.path.join(current_dir, 'util')
        if os.path.exists(potential_util_path):
            sys.path.append(potential_util_path)
            return
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Couldn't find 'util' directory.")

find_and_append_util_path()
import webcamera_utils

#---------------------------
# グローバル変数
#---------------------------
NUM_MODEL_MAX = 16
recv_frames = [None for _ in range(NUM_MODEL_MAX)]
recv_locks  = [threading.Lock() for _ in range(NUM_MODEL_MAX)]
recv_counts = [0 for _ in range(NUM_MODEL_MAX)]   # 受信数
send_counts = [0 for _ in range(NUM_MODEL_MAX)]   # 送信数
stop_flag   = False

# Mutex for count updates
count_lock = threading.Lock()

#---------------------------
# カメラ画像をクライアントに送信するスレッド
#---------------------------
def resize_frame(frame, target_width):
    h, w = frame.shape[:2]
    new_w = target_width
    new_h = int(h * (new_w / w))
    resized = cv2.resize(frame, (new_w, new_h))
    return resized

def send_controlled_thread(writers, cap, n_clients, target_width):
    global stop_flag, recv_counts, send_counts

    print("[Thread] camera capture and controlled send started")
    fps = 20.0

    # 初期的に2枚ずつ送信する
    for i in range(2):
        ret, frame = cap.read()
        if not ret:
            break
        for ci, w in enumerate(writers):
            try:
                w.write(resize_frame(frame, target_width))
                with count_lock:
                    send_counts[ci] += 1
            except Exception as e:
                print(f"[Thread] init send error -> client {ci}: {e}")
        time.sleep(1.0 / fps)
    print("[Thread] initial 2 frames sent to all clients")

    # メイン送信ループ
    while not stop_flag:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # 各クライアントに対して「受信が2フレーム分遅れていない」ものだけ送信
        with count_lock:
            should_send = [recv_counts[i] >= (send_counts[i] - 1) for i in range(n_clients)]

        for ci in range(n_clients):
            if should_send[ci]:  # このクライアントは送信許可
                try:
                    writers[ci].write(resize_frame(frame, target_width))
                    with count_lock:
                        send_counts[ci] += 1
                except Exception as e:
                    print(f"[Thread] send err -> client {ci}: {e}")
        time.sleep(1.0 / fps)

    for w in writers:
        w.release()
    print("[Thread] camera capture and controlled send finished")

#---------------------------
# クライアントから映像を受信するスレッド
#---------------------------
def recv_thread(idx, addr):
    global stop_flag, recv_frames, recv_counts
    print(f"[Thread-{idx}] receiver connecting to {addr}")
    reader = None
    while reader is None:
        try:
            reader = webcamera_utils.get_capture(addr)
        except Exception as e:
            print(f"[Thread-{idx}] retrying connection to {addr} ...")
            time.sleep(1)
    print(f"[Thread-{idx}] connected {addr}")

    while not stop_flag and reader.isOpened():
        ret, frame = reader.read()
        if not ret or frame is None:
            continue
        with recv_locks[idx]:
            recv_frames[idx] = frame
        with count_lock:
            recv_counts[idx] += 1   # 受信完了カウント増やす
    reader.release()
    print(f"[Thread-{idx}] receiver stopped")

#---------------------------
# グリッド表示用関数
#---------------------------
def make_grid(frames, rows=1, cols=3, cell_size=(640,480)):
    canvas = np.zeros((cell_size[1]*rows, cell_size[0]*cols, 3), dtype=np.uint8)
    for i,frame in enumerate(frames):
        if frame is None:
            continue
        rframe = cv2.resize(frame, cell_size)
        rr, cc = divmod(i, cols)
        y0 = rr*cell_size[1]
        x0 = cc*cell_size[0]
        canvas[y0:y0+cell_size[1], x0:x0+cell_size[0]] = rframe
    return canvas

#---------------------------
# メイン
#---------------------------
def main():
    global stop_flag

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_ports', nargs='+', required=True)
    parser.add_argument('-s', '--show_ports',  nargs='+', required=True)
    parser.add_argument('--width',  type=int, default=640, help="送受信するカメラ画像の横幅を指定します。")
    parser.add_argument('--height', type=int, default=480, help="送受信するカメラ画像の高さを指定します。")
    args = parser.parse_args()

    if len(args.video_ports) != len(args.show_ports):
        print("Error: -v と -s の数を一致させてください。")
        return

    n_clients = len(args.video_ports)
    cap = webcamera_utils.get_capture("0")
    if not cap or not cap.isOpened():
        print("Error: カメラが開けません。")
        return

    # Writer作成
    writers = []
    for addr in args.video_ports:
        w = webcamera_utils.get_writer(addr, args.height, args.width, 20)
        writers.append(w)

    # 送信スレッド
    sender_thread = threading.Thread(target=send_controlled_thread,
                                     args=(writers, cap, n_clients, args.width),
                                     daemon=True)
    sender_thread.start()

    # 受信スレッド
    recv_threads_ = []
    for i, addr in enumerate(args.show_ports):
        t = threading.Thread(target=recv_thread, args=(i, addr), daemon=True)
        t.start()
        recv_threads_.append(t)

    # グリッド表示
    print("[Main] displaying grid of results")
    while True:
        frames_copy = []
        for i in range(n_clients):
            with recv_locks[i]:
                frame = recv_frames[i].copy() if recv_frames[i] is not None else None
            frames_copy.append(frame)
        n_row = 2
        n_col = (n_clients + n_row - 1) // n_row
        grid = make_grid(frames_copy, n_col, n_row, (args.width, args.height))
        cv2.imshow("ailia TFLite Runtime Multi Model Inference", grid)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            stop_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Main] all done")

#---------------------------
if __name__ == "__main__":
    main()
