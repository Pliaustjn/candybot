import os
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ================== 配置区（可在 PyCharm 直接修改） ==================
IMAGE_PATH = 'test_screenshot.png'   # 截图路径
WEIGHTS_PATH = 'best.pt'             # YOLO 权重路径
CONFIDENCE = 0.25                    # 置信度阈值
IOU = 0.45                           # NMS IoU 阈值
SAVE_OVERLAY_PATH = 'board_overlay.png'  # 可视化输出路径
# ====================================================================

CLASS_NAMES = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']
EMPTY_VALUE = 6
GRID_SIZE = 9


def detections_to_board(
    boxes_xywhn: np.ndarray,
    classes: np.ndarray,
    grid_size: int = GRID_SIZE,
    empty_value: int = EMPTY_VALUE,
) -> Tuple[List[List[int]], List[Tuple[int, int, int, int]]]:
    """Convert YOLO normalized boxes/classes to NxN board."""
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


def draw_overlay(image: np.ndarray, board: List[List[int]], save_path: str) -> None:
    h, w = image.shape[:2]
    out = image.copy()

    for i in range(1, GRID_SIZE):
        x = int(i * w / GRID_SIZE)
        y = int(i * h / GRID_SIZE)
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)

    cell_w = w / GRID_SIZE
    cell_h = h / GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            val = board[r][c]
            if val == EMPTY_VALUE:
                continue
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            cv2.putText(out, str(val), (cx - 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(save_path, out)
    print(f'\n✓ 已保存可视化结果: {save_path}')


def run(
    image_path: str = IMAGE_PATH,
    weights_path: str = WEIGHTS_PATH,
    conf: float = CONFIDENCE,
    iou: float = IOU,
    save_overlay_path: str = SAVE_OVERLAY_PATH,
) -> List[List[int]]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'输入图片不存在: {image_path}')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'模型文件不存在: {weights_path}')

    model = YOLO(weights_path)
    results = model.predict(source=image_path, conf=conf, iou=iou, verbose=False)
    if not results:
        raise RuntimeError('模型没有返回结果')

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        board = [[EMPTY_VALUE for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        print('⚠️ 没有检测到棋子，返回全空棋盘。')
        print_board(board)
        return board

    boxes_xywhn = result.boxes.xywhn.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    board, conflicts = detections_to_board(boxes_xywhn, classes)

    invalid = sorted({int(c) for c in classes if int(c) < 0 or int(c) >= len(CLASS_NAMES)})
    if invalid:
        print(f'⚠️ 发现未知类别ID: {invalid}（预期 0~5）')

    print_board(board)

    if conflicts:
        print('\n⚠️ 发现格子冲突（多个检测落在同一格，后者覆盖前者）:')
        for row, col, old_cls, new_cls in conflicts:
            print(f'  - cell({row},{col}): {old_cls} -> {new_cls}')

    image = cv2.imread(image_path)
    if image is not None:
        draw_overlay(image, board, save_overlay_path)

    return board


if __name__ == '__main__':
    print('在 PyCharm 直接运行：请先在脚本顶部配置 IMAGE_PATH 和 WEIGHTS_PATH。')
    try:
        run()
    except Exception as exc:
        print(f'❌ 运行失败: {exc}')
