import os
import subprocess
import time
from typing import List, Tuple

import cv2
import numpy as np

PHONE_IP = '192.168.2.16:5555'
TEMP_IMAGE = 'temp_screen.png'
OUTPUT_IMAGE = 'grid_hough_overlay.png'
GRID_SIZE = 8

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


def detect_grid_lines(image: np.ndarray) -> Tuple[List[int], List[int], np.ndarray]:
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

    x_lines = _cluster_positions(xs)
    y_lines = _cluster_positions(ys)
    return x_lines, y_lines, edges


def _fit_to_grid_lines(raw_lines: List[int], size: int, grid_size: int = GRID_SIZE) -> List[int]:
    target = grid_size + 1
    if len(raw_lines) >= target:
        idx = np.linspace(0, len(raw_lines) - 1, target)
        return [int(raw_lines[int(i)]) for i in idx]
    if len(raw_lines) >= 2:
        return [int(v) for v in np.linspace(raw_lines[0], raw_lines[-1], target)]
    return [int(v) for v in np.linspace(int(size * 0.15), int(size * 0.85), target)]


def draw_grid_overlay(image: np.ndarray, x_lines: List[int], y_lines: List[int]) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]

    x_grid = _fit_to_grid_lines(x_lines, w, GRID_SIZE)
    y_grid = _fit_to_grid_lines(y_lines, h, GRID_SIZE)

    for x in x_grid:
        cv2.line(out, (x, 0), (x, h - 1), (0, 255, 255), 2)
    for y in y_grid:
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 255), 2)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cx = int((x_grid[c] + x_grid[c + 1]) / 2)
            cy = int((y_grid[r] + y_grid[r + 1]) / 2)
            cv2.putText(out, f'{r},{c}', (cx - 18, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    return out


def main() -> None:
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

    x_lines, y_lines, edges = detect_grid_lines(image)

    print('\n检测到的候选竖线 x 坐标:', x_lines)
    print('检测到的候选横线 y 坐标:', y_lines)

    overlay = draw_grid_overlay(image, x_lines, y_lines)
    cv2.imwrite(OUTPUT_IMAGE, overlay)
    cv2.imwrite('grid_hough_edges.png', edges)
    print(f'\n✅ 已输出网格叠图: {OUTPUT_IMAGE}')
    print('✅ 已输出边缘图: grid_hough_edges.png')

    if os.path.exists(TEMP_IMAGE):
        os.remove(TEMP_IMAGE)


if __name__ == '__main__':
    main()
