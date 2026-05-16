import os
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

# ================== 配置区（可在 PyCharm 直接修改） ==================
IMAGE_PATH = 'test_screenshot.png'   # 截图路径
WEIGHTS_PATH = 'best.pt'             # YOLO 权重路径
CONFIDENCE = 0.25                    # 置信度阈值
IOU = 0.45                           # NMS IoU 阈值
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


def run(
    image_path: str = IMAGE_PATH,
    weights_path: str = WEIGHTS_PATH,
    conf: float = CONFIDENCE,
    iou: float = IOU,
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

    return board


if __name__ == '__main__':
    print('在 PyCharm 直接运行：请先在脚本顶部配置 IMAGE_PATH 和 WEIGHTS_PATH。')
    print('仅输出9x9棋盘，不保存截图。')
    try:
        run()
    except Exception as exc:
        print(f'❌ 运行失败: {exc}')
