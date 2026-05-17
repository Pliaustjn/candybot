import os
import subprocess
import time
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

# ================== 配置区（可在 PyCharm 直接修改） ==================
PHONE_IP = '192.168.2.16:5555'       # 手机 adb 地址
WEIGHTS_PATH = 'best.pt'             # YOLO 权重路径
CONFIDENCE = 0.25                    # 置信度阈值
IOU = 0.45                           # NMS IoU 阈值
CAPTURE_INTERVAL_SEC = 1.0           # 每次输出棋盘的间隔（秒）
# ====================================================================

CLASS_NAMES = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']
EMPTY_VALUE = 6
GRID_SIZE = 9


class RealtimeBoardDetector:
    def __init__(self, phone_ip: str = PHONE_IP):
        self.phone_ip = phone_ip

    def adb_connect(self) -> bool:
        print(f'🔌 正在连接 {self.phone_ip} ...')
        subprocess.run(['adb', 'connect', self.phone_ip], capture_output=True, timeout=10)
        time.sleep(1)
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=10)
        if self.phone_ip in result.stdout and 'device' in result.stdout:
            print(f'✓ 已连接到 {self.phone_ip}')
            return True
        print('✗ adb 连接失败，请检查手机和端口')
        return False

    def capture_screenshot_to_local(self, local_path: str = 'temp_screen.png') -> bool:
        try:
            subprocess.run(['adb', 'shell', 'screencap', '/sdcard/screen.png'], capture_output=True, timeout=8)
            subprocess.run(['adb', 'pull', '/sdcard/screen.png', local_path], capture_output=True, timeout=8)
            subprocess.run(['adb', 'shell', 'rm', '/sdcard/screen.png'], capture_output=True, timeout=5)
            return os.path.exists(local_path) and os.path.getsize(local_path) > 0
        except Exception as exc:
            print(f'截图失败: {exc}')
            return False


def detections_to_board(
    boxes_xywhn: np.ndarray,
    classes: np.ndarray,
    grid_size: int = GRID_SIZE,
    empty_value: int = EMPTY_VALUE,
) -> Tuple[List[List[int]], List[Tuple[int, int, int, int]]]:
    board = [[empty_value for _ in range(grid_size)] for _ in range(grid_size)]
    conflicts = []

    for box, cls in zip(boxes_xywhn, classes):
        x_center, y_center = float(box[0]), float(box[1])
        col = min(grid_size - 1, max(0, int(x_center * grid_size)))
        row = min(grid_size - 1, max(0, int(y_center * grid_size)))

        cls_int = int(cls)
        if board[row][col] != empty_value:
            conflicts.append((row, col, board[row][col], cls_int))
        board[row][col] = cls_int

    return board, conflicts


def print_board(board: List[List[int]]) -> None:
    print('\n9x9 棋盘（0-5=棋子类别, 6=空）:')
    for row in board:
        print(' '.join(map(str, row)))


def infer_board_from_image(model: YOLO, image_path: str, conf: float, iou: float) -> List[List[int]]:
    results = model.predict(source=image_path, conf=conf, iou=iou, verbose=False)
    if not results:
        raise RuntimeError('模型没有返回结果')

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        board = [[EMPTY_VALUE for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        print('⚠️ 没有检测到棋子，返回全空棋盘。')
        return board

    boxes_xywhn = result.boxes.xywhn.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    board, conflicts = detections_to_board(boxes_xywhn, classes)

    invalid = sorted({int(c) for c in classes if int(c) < 0 or int(c) >= len(CLASS_NAMES)})
    if invalid:
        print(f'⚠️ 发现未知类别ID: {invalid}（预期 0~5）')

    if conflicts:
        print('\n⚠️ 发现格子冲突（多个检测落在同一格，后者覆盖前者）:')
        for row, col, old_cls, new_cls in conflicts:
            print(f'  - cell({row},{col}): {old_cls} -> {new_cls}')

    return board


def run_realtime(
    weights_path: str = WEIGHTS_PATH,
    conf: float = CONFIDENCE,
    iou: float = IOU,
    interval_sec: float = CAPTURE_INTERVAL_SEC,
) -> None:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'模型文件不存在: {weights_path}')

    detector = RealtimeBoardDetector(PHONE_IP)
    if not detector.adb_connect():
        return

    model = YOLO(weights_path)
    temp_image_path = 'temp_screen.png'

    print('\n开始实时识别：每次截图后输出9x9棋盘。')
    print('按 Ctrl+C 结束。')

    try:
        while True:
            ok = detector.capture_screenshot_to_local(temp_image_path)
            if not ok:
                print('⚠️ 当前帧截图失败，下一次重试...')
                time.sleep(interval_sec)
                continue

            board = infer_board_from_image(model, temp_image_path, conf, iou)
            print_board(board)
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print('\n🛑 已停止实时识别。')
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


if __name__ == '__main__':
    print('在 PyCharm 直接运行：实时 adb 截图 -> YOLO 识别 -> 输出9x9棋盘。')
    try:
        run_realtime()
    except Exception as exc:
        print(f'❌ 运行失败: {exc}')
