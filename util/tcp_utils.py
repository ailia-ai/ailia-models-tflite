import cv2
import socket
import struct
import numpy as np


class TcpVideoWriter:
    """Send frames over TCP (encoded with JPEG, with size header)"""
    def __init__(self, addr, height, width, fps=20, rgb=True, quality=95):
        host, port = addr.split(':')
        port = int(port)

        self.host = host
        self.port = port
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        self.height = height
        self.width = width
        self.fps = fps
        self.rgb = rgb
        self.sock = None
        self.isOpened_flag = True
    
    def connect(self):
        """サーバー接続"""
        if self.sock is not None:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[TcpVideoWriter] Connecting to {self.host}:{self.port} ...")
        self.sock.connect((self.host, self.port))
        print("[TcpVideoWriter] Connected.")

    def write(self, frame):
        """Send JPEG + size(header)"""
        self.connect()
        try:
            result, encimg = cv2.imencode(".jpg", frame, self.encode_param)
            if not result:
                print("[TcpVideoWriter] imencode failed.")
                return

            data = encimg.tobytes()
            # 各JPEGデータの前に「サイズ(4バイト)」を送信
            size_header = struct.pack(">L", len(data))
            self.sock.sendall(size_header + data)
        except Exception as e:
            print(f"[TcpVideoWriter] Error sending frame: {e}")
            self.isOpened_flag = False

    def release(self):
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        self.isOpened_flag = False

    def isOpened(self):
        return self.isOpened_flag


class TcpVideoReader:
    """Receive JPEG-encoded frames over TCP (with 4-byte size header)"""
    def __init__(self, addr):
        host, port = addr.split(':')
        port = int(port)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        print(f"[TcpVideoReader] Listening on {host}:{port} ...")

        # クライアント接続待ち
        self.conn, self.client_addr = self.sock.accept()
        print(f"[TcpVideoReader] Connection from {self.client_addr}")

        self.isOpened_flag = True
        self.payload_size = struct.calcsize(">L")  # サイズヘッダは 4バイト

    def _recvall(self, size):
        """size分だけ確実に受信"""
        buf = b""
        while len(buf) < size:
            packet = self.conn.recv(size - len(buf))
            if not packet:
                return None
            buf += packet
        return buf

    def read(self):
        """1フレーム分受信"""
        try:
            # まずJPEGデータサイズ(4B)
            packed_size = self._recvall(self.payload_size)
            if not packed_size:
                self.isOpened_flag = False
                return False, None

            jpeg_size = struct.unpack(">L", packed_size)[0]
            # JPEGデータ本体
            jpeg_data = self._recvall(jpeg_size)
            if not jpeg_data:
                self.isOpened_flag = False
                return False, None

            # JPEG → ndarrayにデコード
            img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                print("[TcpVideoReader] imdecode failed.")
                return False, None
            return True, frame
        except Exception as e:
            print(f"[TcpVideoReader] Error reading frame: {e}")
            self.isOpened_flag = False
            return False, None

    def release(self):
        """接続終了処理"""
        try:
            self.conn.close()
            self.sock.close()
        except:
            pass
        self.isOpened_flag = False

    def isOpened(self):
        return self.isOpened_flag

    def get(self, prop):
        """ダミーのプロパティ値"""
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        if prop == cv2.CAP_PROP_FPS:
            return 30
        return None
