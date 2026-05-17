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
USE_COL_ROW_FOR_EXECUTION = False  # True: 按(col,row)解释RL输出再执行
LAST_ACTION = None
REPEAT_SAME_ACTION = 0
REPEAT_LIMIT = 2
TOPK_SCORE_SEARCH = 20
MIN_SCORE_ACTION = 10

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



def build_action_mask(board: List[List[int]], size: int = GRID_SIZE) -> np.ndarray:
    """1=legal, 0=illegal (swap touching empty cell)."""
    board_np = np.array(board)
    n_actions = (size * (size - 1)) * 2
    mask = np.ones(n_actions, dtype=np.float32)
    for a in range(n_actions):
        (r1, c1), (r2, c2) = decode_action(a, size)
        if board_np[r1, c1] == -1 or board_np[r2, c2] == -1:
            mask[a] = 0.0
    return mask


def find_matches(board: List[List[int]]) -> List[Tuple[int, int]]:
    arr = np.array(board, dtype=int)
    size = arr.shape[0]
    m = set()
    for r in range(size):
        c = 0
        while c < size - 2:
            v = arr[r, c]
            if v == -1:
                c += 1
                continue
            run = 1
            while c + run < size and arr[r, c + run] == v:
                run += 1
            if run >= 3:
                for k in range(run):
                    m.add((r, c + k))
            c += run
    for c in range(size):
        r = 0
        while r < size - 2:
            v = arr[r, c]
            if v == -1:
                r += 1
                continue
            run = 1
            while r + run < size and arr[r + run, c] == v:
                run += 1
            if run >= 3:
                for k in range(run):
                    m.add((r + k, c))
            r += run
    return sorted(m)


def estimate_action_score(board: List[List[int]], action: int) -> int:
    (r1, c1), (r2, c2) = decode_action(action, GRID_SIZE)
    b = [row[:] for row in board]
    b[r1][c1], b[r2][c2] = b[r2][c2], b[r1][c1]
    matches = find_matches(b)
    if not matches:
        return 0
    score = len(matches) * 10
    if len(matches) >= 4:
        score += 50
    if len(matches) >= 5:
        score += 100
    return score

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







def maybe_swap_rc_for_execution(r: int, c: int) -> Tuple[int, int]:
    """Optionally treat RL output as (col,row) for execution experiments."""
    if USE_COL_ROW_FOR_EXECUTION:
        return c, r
    return r, c

def rc_to_xy(r: int, c: int) -> Tuple[int, int]:
    """Convert internal (row,col) to user-facing (x,y)."""
    return c, r

def align_endpoint_by_action(sx: int, sy: int, tx: int, ty: int, dr: int, dc: int) -> Tuple[int, int]:
    """Force endpoint to follow decoded action axis and sign exactly."""
    if dr != 0 and dc == 0:
        tx = sx
        if dr > 0 and ty < sy:
            ty = sy + abs(ty - sy)
        elif dr < 0 and ty > sy:
            ty = sy - abs(ty - sy)
    elif dc != 0 and dr == 0:
        ty = sy
        if dc > 0 and tx < sx:
            tx = sx + abs(tx - sx)
        elif dc < 0 and tx > sx:
            tx = sx - abs(tx - sx)
    return tx, ty

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
    global LAST_ACTION, REPEAT_SAME_ACTION
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
    state_dict = torch.load(RL_MODEL_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    # 强制仅决策模式：关闭训练态 + 冻结参数梯度
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    torch.set_grad_enabled(False)

    with torch.inference_mode():
        probs, _ = model(torch.FloatTensor(obs).unsqueeze(0))
        prob_vec = probs.squeeze(0).cpu().numpy()

    # 输出RL原始策略Top-5（未掩码）
    raw_topk = np.argsort(-prob_vec)[:5]
    print('RL原始输出Top-5(未掩码):')
    for rk, a in enumerate(raw_topk, 1):
        a = int(a)
        (rr1, cc1), (rr2, cc2) = decode_action(a, GRID_SIZE)
        print(f'  {rk}. action={a}, prob={prob_vec[a]:.6f}, swap(row,col)=({rr1},{cc1})<->({rr2},{cc2})')

    # 动作掩码：过滤掉涉及空位(-1)的非法交换，再做argmax
    mask = build_action_mask(board, GRID_SIZE)
    masked = prob_vec * mask
    if masked.sum() <= 0:
        # 兜底：若全被mask掉，退回原始argmax
        action = int(np.argmax(prob_vec))
        print('⚠️ 动作掩码后无可用动作，回退原始argmax')
    else:
        action = int(np.argmax(masked))

    # 在Top-K合法动作里优先挑可直接得分动作
    ranked = np.argsort(-masked)
    best_action = action
    best_score = estimate_action_score(board, action) if masked[action] > 0 else 0
    for cand in ranked[:TOPK_SCORE_SEARCH]:
        cand = int(cand)
        if masked[cand] <= 0:
            continue
        sc = estimate_action_score(board, cand)
        if sc > best_score:
            best_score = sc
            best_action = cand
    if best_action != action and best_score >= MIN_SCORE_ACTION:
        print(f'✅ 评分重排: action {action} -> {best_action}, score {best_score}')
        action = best_action
    elif best_score <= 0:
        print('⚠️ 当前Top合法动作都无法直接得分')

    rl_mode = 'eval' if not model.training else 'train'
    print(f'RL model mode: {rl_mode} (decision-only)')
    (r1, c1), (r2, c2) = decode_action(action, GRID_SIZE)

    if LAST_ACTION == action:
        REPEAT_SAME_ACTION += 1
    else:
        REPEAT_SAME_ACTION = 0
    if REPEAT_SAME_ACTION >= REPEAT_LIMIT:
        ranked_masked = np.argsort(-masked)
        changed = False
        for cand in ranked_masked:
            cand = int(cand)
            if cand == action or masked[cand] <= 0:
                continue
            action = cand
            (r1, c1), (r2, c2) = decode_action(action, GRID_SIZE)
            REPEAT_SAME_ACTION = 0
            changed = True
            print(f'⚠️ 检测到重复动作，切换次优合法动作: {action}')
            break
        if not changed:
            print('⚠️ 重复动作但未找到次优合法动作')
    LAST_ACTION = action

    legal_count = int(mask.sum())
    print(f'RL 动作: {action}, swap(row,col)=({r1},{c1}) <-> ({r2},{c2}) | legal_actions={legal_count}')

    er1, ec1 = maybe_swap_rc_for_execution(r1, c1)
    er2, ec2 = maybe_swap_rc_for_execution(r2, c2)
    exec_mode = 'col,row' if USE_COL_ROW_FOR_EXECUTION else 'row,col'
    print(f'执行坐标模式: {exec_mode}')
    print(f'执行格子(row,col): ({er1},{ec1}) <-> ({er2},{ec2})')

    # 空位保护：如果动作落在空位则不执行
    if board[er1][ec1] == -1 or board[er2][ec2] == -1:
        print('⚠️ RL 动作包含空位，取消执行 adb 移动')
        return

    # 优先使用 YOLO 实际检测到的格子中心（更贴近真实棋子位置）
    if (er1, ec1) in cell_center_map:
        sx, sy = cell_center_map[(er1, ec1)]
    else:
        sx, sy = cell_center(x0, x1, y0, y1, er1, ec1)

    if (er2, ec2) in cell_center_map:
        tx, ty = cell_center_map[(er2, ec2)]
    else:
        # 回退：按动作方向构造终点
        cell_w = max(1.0, (x1 - x0) / GRID_SIZE)
        cell_h = max(1.0, (y1 - y0) / GRID_SIZE)
        dx = ec2 - ec1
        dy = er2 - er1
        tx = int(sx + dx * cell_w * 0.8)
        ty = int(sy + dy * cell_h * 0.8)

    dr, dc = (er2 - er1), (ec2 - ec1)
    tx, ty = align_endpoint_by_action(sx, sy, tx, ty, dr, dc)

    dev_w, dev_h = get_device_screen_size()
    dsx, dsy = map_img_to_device(sx, sy, w, h, dev_w, dev_h)
    dtx, dty = map_img_to_device(tx, ty, w, h, dev_w, dev_h)

    actual_dr = 'down' if ty > sy else ('up' if ty < sy else 'none')
    actual_dc = 'right' if tx > sx else ('left' if tx < sx else 'none')
    print(f'adb swipe(img): ({sx},{sy}) -> ({tx},{ty}) | decoded_from_exec=(dx={dc},dy={dr}) actual=({actual_dc},{actual_dr})')
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
