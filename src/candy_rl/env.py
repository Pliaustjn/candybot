from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CandyCrushEnv(gym.Env):
    """Gymnasium-compatible Candy Crush-like environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size: int = 8, n_colors: int = 6, max_steps: int = 100, hole_ratio: float = 0.08):
        super().__init__()
        self.board_size = board_size
        self.n_colors = n_colors
        self.max_steps = max_steps
        self.hole_ratio = hole_ratio

        self.board: np.ndarray | None = None
        self.score = 0
        self.steps = 0

        self.n_actions = (board_size * (board_size - 1)) * 2
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_colors + 1, self.board_size, self.board_size),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.board = self.np_random.integers(0, self.n_colors, (self.board_size, self.board_size), endpoint=False)
        self.score = 0
        self.steps = 0

        while self._has_matches():
            self._clear_all_matches()
            self._apply_gravity()
            self._refill()

        self._inject_holes(self.hole_ratio)
        return self._get_observation(), {"score": self.score}

    def step(self, action: int):
        self.steps += 1
        pos1, pos2 = self._decode_action(action)

        if self._is_valid_swap(pos1, pos2):
            self._swap(pos1, pos2)
            if self._has_matches():
                round_score = self._process_chain_reaction()
                self.score += round_score
                reward = float(round_score)
            else:
                self._swap(pos1, pos2)
                reward = -1.0
        else:
            reward = -2.0

        terminated = self.steps >= self.max_steps
        truncated = False
        return self._get_observation(), reward, terminated, truncated, {"score": self.score}

    def _inject_holes(self, hole_ratio: float = 0.08) -> None:
        total = self.board_size * self.board_size
        n_holes = int(total * hole_ratio)
        if n_holes <= 0:
            return
        indices = self.np_random.choice(total, size=n_holes, replace=False)
        rows = indices // self.board_size
        cols = indices % self.board_size
        self.board[rows, cols] = -1

    def _get_observation(self):
        one_hot = np.zeros((self.n_colors + 1, self.board_size, self.board_size), dtype=np.float32)
        for color in range(self.n_colors):
            one_hot[color] = (self.board == color).astype(np.float32)
        one_hot[self.n_colors] = (self.board == -1).astype(np.float32)
        return one_hot

    def _decode_action(self, action: int):
        size = self.board_size
        if action < size * (size - 1):
            row = action // (size - 1)
            col = action % (size - 1)
            return (row, col), (row, col + 1)

        action -= size * (size - 1)
        row = action // size
        col = action % size
        return (row, col), (row + 1, col)

    def _is_valid_swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        size = self.board_size
        if not (0 <= r1 < size and 0 <= c1 < size and 0 <= r2 < size and 0 <= c2 < size):
            return False
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
        for col in range(size):
            for row in range(size - 2):
                v = self.board[row, col]
                if v != -1 and v == self.board[row + 1, col] == self.board[row + 2, col]:
                    matches.update([(row, col), (row + 1, col), (row + 2, col)])
        return list(matches)

    def _clear_matches(self, matches):
        for row, col in matches:
            self.board[row, col] = -1

    def _clear_all_matches(self):
        matches = self._find_all_matches()
        self._clear_matches(matches)

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
                    self.board[row, col] = self.np_random.integers(0, self.n_colors)

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
