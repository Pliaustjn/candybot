import os
import subprocess
import time
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

PHONE_IP = '192.168.2.16:5555'
TEMP_IMAGE = 'temp_screen.png'
WEIGHTS_PATH = 'best.pt'
CONFIDENCE = 0.25
IOU = 0.45
GRID_SIZE = 8

CLASS_NAMES = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']

CANNY_LOW = 60
CANNY_HIGH = 180
HOUGH_THRESHOLD = 120
MIN_LINE_LENGTH = 200
MAX_LINE_GAP = 20


class AdbCapture:
    def __init__(self, phone_ip: str = PHONE_IP):
        self.phone_ip = phone_ip

    def connect(self) -> bool:
        subprocess.run(['adb', 'connect', self.phone_ip], capture_output=True, timeout=10)
        time.sleep(1)
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=10)
        return self.phone_ip in result.stdout and 'device' in result.stdout

    def capture(self, local_path: str = TEMP_IMAGE) -> bool:
        try:
            subprocess.run(['adb', 'shell', 'screencap', '/sdcard/screen.png'], capture_output=True, timeout=8)
            subprocess.run(['adb', 'pull', '/sdcard/screen.png', local_path], capture_output=True, timeout=8)
            subprocess.run(['adb', 'shell', 'rm', '/sdcard/screen.png'], capture_output=True, timeout=5)
            return os.path.exists(local_path) and os.path.getsize(local_path) > 0
        except Exception:
            return False


def _cluster_positions(vals: List[int], tol: int = 12) -> List[int]:
    if not vals:
        return []
    vals = sorted(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(np.mean(g)) for g in groups]


def detect_grid_lines(image: np.ndarray) -> Tuple[List[int], List[int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    h, w = image.shape[:2]
    xs, ys = [], []
    if lines is not None:
        for ln in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, ln)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx < 8 and dy > h * 0.18:
                xs.append((x1 + x2) // 2)
            elif dy < 8 and dx > w * 0.18:
                ys.append((y1 + y2) // 2)

    return _cluster_positions(xs), _cluster_positions(ys)


def _middle_bounds(lines: List[int], size: int) -> Tuple[int, int]:
    arr = np.array(sorted(lines), dtype=float)
    if arr.size >= 4:
        low = int(np.percentile(arr, 20))
        high = int(np.percentile(arr, 80))
    elif arr.size >= 2:
        low, high = int(arr.min()), int(arr.max())
    else:
        low, high = int(size * 0.2), int(size * 0.8)
    return max(0, low), min(size, high)


def _middle_bounds_y(lines: List[int], size: int) -> Tuple[int, int]:
    arr = np.array(sorted(lines), dtype=float)
    if arr.size >= 6:
        low = int(arr[4])
        high = int(arr[5])
    else:
        low, high = _middle_bounds(lines, size)
    return max(0, low), min(size, high)


def map_detections_to_8x8(result, img_w: int, img_h: int, x0: int, x1: int, y0: int, y1: int):
    pieces = []
    for xywhn, cls, conf in zip(result.boxes.xywhn.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        cx = float(xywhn[0]) * img_w
        cy = float(xywhn[1]) * img_h
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        col = int((cx - x0) / max(1e-6, (x1 - x0)) * GRID_SIZE)
        row = int((cy - y0) / max(1e-6, (y1 - y0)) * GRID_SIZE)
        col = min(GRID_SIZE - 1, max(0, col))
        row = min(GRID_SIZE - 1, max(0, row))
        cls_i = int(cls)
        name = CLASS_NAMES[cls_i] if 0 <= cls_i < len(CLASS_NAMES) else f'unknown_{cls_i}'
        pieces.append((row, col, cls_i, name, float(conf), int(cx), int(cy)))
    return pieces



def build_board_array(pieces, grid_size: int = GRID_SIZE, empty_value: int = -1):
    board = [[empty_value for _ in range(grid_size)] for _ in range(grid_size)]
    # 同一格多个棋子时，保留置信度更高者（pieces 已按 row,col,-conf 排序）
    for row, col, cls_i, _name, _conf, _cx, _cy in pieces:
        if board[row][col] == empty_value:
            board[row][col] = int(cls_i)
    return board

def main() -> None:
    if not os.path.exists(WEIGHTS_PATH):
        print(f'❌ 模型文件不存在: {WEIGHTS_PATH}')
        return

    cap = AdbCapture(PHONE_IP)
    print(f'连接 adb: {PHONE_IP}')
    if not cap.connect():
        print('❌ adb 连接失败')
        return

    print('截图中...')
    if not cap.capture(TEMP_IMAGE):
        print('❌ 截图失败')
        return

    image = cv2.imread(TEMP_IMAGE)
    if image is None:
        print('❌ 无法读取截图')
        return

    h, w = image.shape[:2]
    x_lines, y_lines = detect_grid_lines(image)
    print('候选竖线 x 坐标:', x_lines)
    print('候选横线 y 坐标:', y_lines)

    x0, x1 = _middle_bounds(x_lines, w)
    y0, y1 = _middle_bounds_y(y_lines, h)
    print(f'使用的8x8棋盘区域: x=[{x0},{x1}], y=[{y0},{y1}]')

    model = YOLO(WEIGHTS_PATH)
    results = model.predict(source=TEMP_IMAGE, conf=CONFIDENCE, iou=IOU, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print('⚠️ 未检测到棋子')
        return

    pieces = map_detections_to_8x8(results[0], w, h, x0, x1, y0, y1)
    pieces = sorted(pieces, key=lambda x: (x[0], x[1], -x[4]))

    print('\n棋子位置（row, col, class_id, class_name, conf, cx, cy）:')
    for p in pieces:
        print(p)

    board = build_board_array(pieces, GRID_SIZE, -1)
    print('\n8x8棋盘整数数组（class_id，空位=-1）:')
    for row in board:
        print(row)

    if os.path.exists(TEMP_IMAGE):
        os.remove(TEMP_IMAGE)


if __name__ == '__main__':
    main()
