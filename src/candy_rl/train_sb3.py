from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from candy_rl.env import CandyCrushEnv


@dataclass
class Config:
    total_timesteps: int
    n_envs: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    run_root: str
    seed: int


class MetricsCallback(BaseCallback):
    def __init__(self, run_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.csv_path = run_dir / "metrics.csv"
        self.jsonl_path = run_dir / "metrics.jsonl"
        self.rows = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            ep = self.model.ep_info_buffer[-1]
            row = {
                "timesteps": int(self.num_timesteps),
                "ep_reward": float(ep.get("r", 0.0)),
                "ep_length": float(ep.get("l", 0.0)),
                "learning_rate": float(self.model.lr_schedule(1.0)),
            }
            self.rows.append(row)
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def make_run_dir(root: str) -> Path:
    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(config_path: str = "configs/train.yaml") -> None:
    cfg = load_config(config_path)
    run_dir = make_run_dir(cfg.run_root)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Run directory: %s", run_dir)

    env = make_vec_env(CandyCrushEnv, n_envs=cfg.n_envs, seed=cfg.seed)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tensorboard_log=str(run_dir / "tb"),
        verbose=1,
        seed=cfg.seed,
    )

    callback = MetricsCallback(run_dir=run_dir)
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback, progress_bar=False)

    model.save(str(run_dir / "ppo_candy_model"))
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.__dict__, f, sort_keys=False)

    logging.info("Training finished. Model saved to %s", run_dir)


if __name__ == "__main__":
    main()
