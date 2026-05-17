import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output
import matplotlib.pyplot as plt


class CandyCrushEnv:
    """Candy Crush 环境：支持空位(-1)，且禁止从空位发起移动。"""

    def __init__(self, board_size=8, n_colors=6):
        self.board_size = board_size
        self.n_colors = n_colors
        self.board = None
        self.score = 0
        self.steps = 0
        self.max_steps = 100

        # 每个格子：向右/向下
        self.n_actions = (board_size * (board_size - 1)) * 2

    def reset(self):
        self.board = np.random.randint(0, self.n_colors, (self.board_size, self.board_size))
        self.score = 0
        self.steps = 0

        while self._has_matches():
            self._clear_all_matches()
            self._apply_gravity()
            self._refill()

        # 额外随机挖空一部分格子，模拟障碍/空洞
        self._inject_holes(hole_ratio=0.08)
        return self._get_observation()

    def _inject_holes(self, hole_ratio=0.08):
        total = self.board_size * self.board_size
        n_holes = int(total * hole_ratio)
        if n_holes <= 0:
            return
        idx = np.random.choice(total, size=n_holes, replace=False)
        r = idx // self.board_size
        c = idx % self.board_size
        self.board[r, c] = -1

    def step(self, action):
        self.steps += 1
        pos1, pos2 = self._decode_action(action)

        if self._is_valid_swap(pos1, pos2):
            self._swap(pos1, pos2)
            if self._has_matches():
                round_score = self._process_chain_reaction()
                self.score += round_score
                reward = round_score
            else:
                self._swap(pos1, pos2)
                reward = -1
        else:
            reward = -2  # 非法动作更强惩罚

        done = self.steps >= self.max_steps
        return self._get_observation(), reward, done, {'score': self.score}

    def _get_observation(self):
        # 6色 + 1个empty通道
        one_hot = np.zeros((self.n_colors + 1, self.board_size, self.board_size), dtype=np.float32)
        for color in range(self.n_colors):
            one_hot[color] = (self.board == color).astype(np.float32)
        one_hot[self.n_colors] = (self.board == -1).astype(np.float32)
        return one_hot

    def _decode_action(self, action):
        size = self.board_size
        if action < size * (size - 1):
            row = action // (size - 1)
            col = action % (size - 1)
            return (row, col), (row, col + 1)
        action -= size * (size - 1)
        row = action // size
        col = action % size
        if row < size - 1:
            return (row, col), (row + 1, col)
        return (0, 0), (0, 0)

    def _is_valid_swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        size = self.board_size
        if not (0 <= r1 < size and 0 <= c1 < size and 0 <= r2 < size and 0 <= c2 < size):
            return False

        # 新规则：空位不能移动（起点/终点任一为空都不允许交换）
        if self.board[r1, c1] == -1 or self.board[r2, c2] == -1:
            return False

        return True

    def _swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]

    def _has_matches(self):
        return len(self._find_all_matches()) > 0

    def _find_all_matches(self):
        size = self.board_size
        matches = set()
        for row in range(size):
            for col in range(size - 2):
                v = self.board[row, col]
                if v != -1 and v == self.board[row, col + 1] == self.board[row, col + 2]:
                    matches.update([(row, col), (row, col + 1), (row, col + 2)])
                    for k in range(3, size - col):
                        if self.board[row, col + k] == v:
                            matches.add((row, col + k))
                        else:
                            break
        for col in range(size):
            for row in range(size - 2):
                v = self.board[row, col]
                if v != -1 and v == self.board[row + 1, col] == self.board[row + 2, col]:
                    matches.update([(row, col), (row + 1, col), (row + 2, col)])
                    for k in range(3, size - row):
                        if self.board[row + k, col] == v:
                            matches.add((row + k, col))
                        else:
                            break
        return list(matches)

    def _clear_matches(self, matches):
        for row, col in matches:
            self.board[row, col] = -1

    def _clear_all_matches(self):
        matches = self._find_all_matches()
        self._clear_matches(matches)
        return len(matches)

    def _apply_gravity(self):
        size = self.board_size
        for col in range(size):
            column = [self.board[row, col] for row in range(size) if self.board[row, col] != -1]
            for row in range(size - 1, -1, -1):
                self.board[row, col] = column.pop() if column else -1

    def _refill(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == -1:
                    self.board[row, col] = np.random.randint(0, self.n_colors)

    def _process_chain_reaction(self):
        total_score = 0
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            score = len(matches) * 10
            if len(matches) >= 4:
                score += 50
            if len(matches) >= 5:
                score += 100
            total_score += score
            self._clear_matches(matches)
            self._apply_gravity()
            self._refill()
        return total_score


# 其余 ActorCritic / PPOTrainer 结构可复用你原始代码。
