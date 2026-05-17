import os
import subprocess
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

PHONE_IP = '192.168.2.16:5555'
TEMP_IMAGE = 'temp_screen.png'
WEIGHTS_PATH = 'best.pt'
RL_MODEL_PATH = 'candy_crush_model.pth'
CONFIDENCE = 0.25
IOU = 0.45
GRID_SIZE = 8
CLASS_NAMES = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']

# adb 滑动时长（毫秒）
SWIPE_MS = 120
MOVE_INTERVAL_SEC = 0.8
MAX_MOVES = 30
ACTION_VIS_PATH = 'last_action_vis.png'

CANNY_LOW = 60
CANNY_HIGH = 180
HOUGH_THRESHOLD = 120
MIN_LINE_LENGTH = 200
MAX_LINE_GAP = 20


class ActorCritic(nn.Module):
    def __init__(self, board_size=8, n_channels=7, n_actions=112):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        flatten_size = 64 * board_size * board_size
        self.actor = nn.Sequential(
            nn.Linear(flatten_size, 256), nn.ReLU(),
            nn.Linear(256, n_actions), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(flatten_size, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        f = self.conv_layers(x)
        f = f.view(f.size(0), -1)
        return self.actor(f), self.critic(f)


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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)

    h, w = image.shape[:2]
    xs, ys = [], []
    if lines is not None:
        for ln in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, ln)
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx < 8 and dy > h * 0.18:
                xs.append((x1 + x2) // 2)
            elif dy < 8 and dx > w * 0.18:
                ys.append((y1 + y2) // 2)
    return _cluster_positions(xs), _cluster_positions(ys)


def _middle_bounds(lines: List[int], size: int) -> Tuple[int, int]:
    arr = np.array(sorted(lines), dtype=float)
    if arr.size >= 4:
        low, high = int(np.percentile(arr, 20)), int(np.percentile(arr, 80))
    elif arr.size >= 2:
        low, high = int(arr.min()), int(arr.max())
    else:
        low, high = int(size * 0.2), int(size * 0.8)
    return max(0, low), min(size, high)


def _middle_bounds_y(lines: List[int], size: int) -> Tuple[int, int]:
    arr = np.array(sorted(lines), dtype=float)
    if arr.size >= 6:
        low, high = int(arr[4]), int(arr[5])
    else:
        low, high = _middle_bounds(lines, size)
    return max(0, low), min(size, high)


def board_to_onehot(board: List[List[int]], n_colors: int = 6) -> np.ndarray:
    b = np.array(board, dtype=np.int64)
    one_hot = np.zeros((n_colors + 1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for c in range(n_colors):
        one_hot[c] = (b == c).astype(np.float32)
    one_hot[n_colors] = (b == -1).astype(np.float32)
    return one_hot


def decode_action(action: int, size: int = GRID_SIZE) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if action < size * (size - 1):
        r = action // (size - 1)
        c = action % (size - 1)
        return (r, c), (r, c + 1)
    action -= size * (size - 1)
    r = action // size
    c = action % size
    return (r, c), (r + 1, c)


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
        pieces.append((row, col, int(cls), float(conf), int(cx), int(cy)))
    return sorted(pieces, key=lambda x: (x[0], x[1], -x[3]))


def build_board_array(pieces, empty_value=-1):
    board = [[empty_value for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for row, col, cls_i, _conf, _cx, _cy in pieces:
        if board[row][col] == empty_value:
            board[row][col] = cls_i
    return board




def build_cell_center_map(pieces):
    """Use YOLO piece centers as physical cell centers to reduce grid-fit drift."""
    cell_points = {}
    for row, col, _cls, conf, cx, cy in pieces:
        key = (row, col)
        if key not in cell_points or conf > cell_points[key][0]:
            cell_points[key] = (conf, cx, cy)
    return {k: (v[1], v[2]) for k, v in cell_points.items()}

def print_board_pretty(board: List[List[int]]) -> None:
    print('8x8棋盘视图（空位=-1）:')
    header = '    ' + ' '.join([f'{c:>3}' for c in range(GRID_SIZE)])
    print(header)
    print('    ' + '---' * GRID_SIZE)
    for r, row in enumerate(board):
        row_str = ' '.join([f'{v:>3}' for v in row])
        print(f'{r:>2} | {row_str}')

def cell_center(x0, x1, y0, y1, row, col):
    cw = (x1 - x0) / GRID_SIZE
    ch = (y1 - y0) / GRID_SIZE
    x = int(x0 + (col + 0.5) * cw)
    y = int(y0 + (row + 0.5) * ch)
    return x, y




def save_action_visualization(image: np.ndarray, sx: int, sy: int, tx: int, ty: int, path: str = ACTION_VIS_PATH) -> str:
    vis = image.copy()
    cv2.arrowedLine(vis, (sx, sy), (tx, ty), (0, 0, 255), 4, tipLength=0.2)
    cv2.circle(vis, (sx, sy), 10, (255, 0, 0), -1)
    cv2.circle(vis, (tx, ty), 10, (0, 255, 0), -1)
    cv2.putText(vis, 'START', (sx + 12, sy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(vis, 'END', (tx + 12, ty - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(path, vis)
    return path

def get_device_screen_size() -> Tuple[int, int]:
    """Return (width, height) from `adb shell wm size`, fallback (0,0)."""
    try:
        out = subprocess.run(['adb', 'shell', 'wm', 'size'], capture_output=True, text=True, timeout=5)
        text = out.stdout.strip()
        # e.g. Physical size: 1080x2400
        for part in text.split():
            if 'x' in part and part.replace('x', '').replace(':', '').isdigit() is False:
                pass
        import re
        m = re.search(r'(\d+)x(\d+)', text)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return 0, 0


def map_img_to_device(x: int, y: int, img_w: int, img_h: int, dev_w: int, dev_h: int) -> Tuple[int, int]:
    if dev_w <= 0 or dev_h <= 0 or img_w <= 0 or img_h <= 0:
        return x, y
    return int(x * dev_w / img_w), int(y * dev_h / img_h)

def adb_swipe(x1, y1, x2, y2):
    subprocess.run(['adb', 'shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2), str(SWIPE_MS)], check=False)


def run_once(step_idx: int = 0):
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(RL_MODEL_PATH):
        print('❌ 缺少 best.pt 或 candy_crush_model.pth')
        return

    cap = AdbCapture(PHONE_IP)
    if not cap.connect() or not cap.capture(TEMP_IMAGE):
        print('❌ adb 连接或截图失败')
        return

    image = cv2.imread(TEMP_IMAGE)
    if image is None:
        print('❌ 图片读取失败')
        return
    h, w = image.shape[:2]

    x_lines, y_lines = detect_grid_lines(image)
    x0, x1 = _middle_bounds(x_lines, w)
    y0, y1 = _middle_bounds_y(y_lines, h)

    yolo = YOLO(WEIGHTS_PATH)
    # 强制 YOLO 仅推理模式
    if hasattr(yolo, 'model') and yolo.model is not None:
        yolo.model.eval()
        for p in yolo.model.parameters():
            p.requires_grad = False
    results = yolo.predict(source=TEMP_IMAGE, conf=CONFIDENCE, iou=IOU, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print('⚠️ 未检测到棋子')
        return

    print(f"YOLO model mode: {'eval' if hasattr(yolo, 'model') and (not yolo.model.training) else 'train'} (inference-only)")
    pieces = map_detections_to_8x8(results[0], w, h, x0, x1, y0, y1)
    cell_center_map = build_cell_center_map(pieces)
    board = build_board_array(pieces, -1)
    print(f'\n===== 实际操作 Step {step_idx} =====')
    print('8x8棋盘整数数组:')
    print(board)
    print_board_pretty(board)

    obs = board_to_onehot(board)  # (7,8,8)
    model = ActorCritic(board_size=8, n_channels=7, n_actions=112)
    state_dict = torch.load(RL_MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)

    # 强制仅决策模式：关闭训练态 + 冻结参数梯度
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    torch.set_grad_enabled(False)

    with torch.inference_mode():
        probs, _ = model(torch.FloatTensor(obs).unsqueeze(0))
        action = int(torch.argmax(probs, dim=1).item())

    rl_mode = 'eval' if not model.training else 'train'
    print(f'RL model mode: {rl_mode} (decision-only)')
    (r1, c1), (r2, c2) = decode_action(action, GRID_SIZE)
    print(f'RL 动作: {action}, swap ({r1},{c1}) <-> ({r2},{c2})')

    # 空位保护：如果动作落在空位则不执行
    if board[r1][c1] == -1 or board[r2][c2] == -1:
        print('⚠️ RL 动作包含空位，取消执行 adb 移动')
        return

    # 优先使用 YOLO 实际检测到的格子中心（更贴近真实棋子位置）
    if (r1, c1) in cell_center_map:
        sx, sy = cell_center_map[(r1, c1)]
    else:
        sx, sy = cell_center(x0, x1, y0, y1, r1, c1)

    if (r2, c2) in cell_center_map:
        tx, ty = cell_center_map[(r2, c2)]
    else:
        # 回退：按动作方向构造终点
        cell_w = max(1.0, (x1 - x0) / GRID_SIZE)
        cell_h = max(1.0, (y1 - y0) / GRID_SIZE)
        dx = c2 - c1
        dy = r2 - r1
        tx = int(sx + dx * cell_w * 0.8)
        ty = int(sy + dy * cell_h * 0.8)

    # 关键修复：垂直/水平动作强制轴对齐，避免轻微斜线被游戏识别成错误方向
    dr, dc = (r2 - r1), (c2 - c1)
    if dr != 0 and dc == 0:
        tx = sx
    elif dc != 0 and dr == 0:
        ty = sy

    dev_w, dev_h = get_device_screen_size()
    dsx, dsy = map_img_to_device(sx, sy, w, h, dev_w, dev_h)
    dtx, dty = map_img_to_device(tx, ty, w, h, dev_w, dev_h)

    print(f'adb swipe(img): ({sx},{sy}) -> ({tx},{ty}) | dir=(dr={dr}, dc={dc})')
    print(f'adb swipe(dev): ({dsx},{dsy}) -> ({dtx},{dty}) | wm={dev_w}x{dev_h}')
    vis_path = save_action_visualization(image, sx, sy, tx, ty, ACTION_VIS_PATH)
    print(f'动作可视化图: {vis_path}')
    adb_swipe(dsx, dsy, dtx, dty)
    print('✅ 已发送 adb 移动命令')
    return True


def main():
    print('开始实际ADB操作模式（循环执行）')
    success = 0
    for i in range(1, MAX_MOVES + 1):
        ok = run_once(i)
        if ok:
            success += 1
        time.sleep(MOVE_INTERVAL_SEC)
    print(f'\n完成。共发送 {success}/{MAX_MOVES} 次移动命令。')


if __name__ == '__main__':
    main()
