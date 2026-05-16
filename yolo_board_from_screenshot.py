import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']
EMPTY_VALUE = 6
GRID_SIZE = 9


def detections_to_board(
    boxes_xywhn: np.ndarray,
    classes: np.ndarray,
    grid_size: int = GRID_SIZE,
    empty_value: int = EMPTY_VALUE,
) -> Tuple[List[List[int]], List[Tuple[int, int, int, int]]]:
    """
    Convert YOLO normalized boxes/classes to NxN board.

    Returns:
        board: grid_size x grid_size matrix
        conflicts: list of (row, col, old_cls, new_cls)
    """
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

    # Draw grid
    for i in range(1, GRID_SIZE):
        x = int(i * w / GRID_SIZE)
        y = int(i * h / GRID_SIZE)
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)

    # Put board values at cell centers
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Use YOLO best.pt to detect pieces and build a 9x9 board.')
    parser.add_argument('--image', required=True, help='Input screenshot path')
    parser.add_argument('--weights', default='best.pt', help='YOLO model path (default: best.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--save-overlay', default='board_overlay.png', help='Overlay output image path')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f'❌ 输入图片不存在: {args.image}')
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f'❌ 模型文件不存在: {args.weights}')
        sys.exit(1)

    model = YOLO(args.weights)
    results = model.predict(source=args.image, conf=args.conf, iou=args.iou, verbose=False)

    if not results:
        print('❌ 模型没有返回结果')
        sys.exit(1)

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        board = [[EMPTY_VALUE for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        print('⚠️ 没有检测到棋子，返回全空棋盘。')
        print_board(board)
        return

    boxes_xywhn = result.boxes.xywhn.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    board, conflicts = detections_to_board(boxes_xywhn, classes)

    # Basic class-id validation
    invalid = sorted({int(c) for c in classes if int(c) < 0 or int(c) >= len(CLASS_NAMES)})
    if invalid:
        print(f'⚠️ 发现未知类别ID: {invalid}（预期 0~5）')

    print_board(board)

    if conflicts:
        print('\n⚠️ 发现格子冲突（多个检测落在同一格，后者覆盖前者）:')
        for row, col, old_cls, new_cls in conflicts:
            print(f'  - cell({row},{col}): {old_cls} -> {new_cls}')

    image = cv2.imread(args.image)
    if image is not None:
        draw_overlay(image, board, args.save_overlay)


if __name__ == '__main__':
    main()
