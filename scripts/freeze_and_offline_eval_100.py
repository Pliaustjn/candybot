from __future__ import annotations

"""Freeze best checkpoint as production candidate + offline 100-episode evaluation.

Usage (Colab):
  PYTHONPATH=/content/drive/MyDrive/candybot/src \
  python scripts/freeze_and_offline_eval_100.py
"""

from pathlib import Path
import json
import shutil

import numpy as np
from stable_baselines3 import PPO

from candy_rl.env import CandyCrushEnv


ROOT = Path("/content/drive/MyDrive/candybot")
LOG_ROOT = ROOT / "logs"
BACKUP_ROOT = Path("/content/drive/MyDrive/candy_backups")
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

# 强制指定当前最强候选（按你近期记录）
SOURCE_RUN_NAME = "run_20260519_132905"
SOURCE_CKPT = LOG_ROOT / SOURCE_RUN_NAME / "ppo_candy_model.zip"

# 生产冻结目录
FROZEN_DIR = BACKUP_ROOT / f"production_candidate_{SOURCE_RUN_NAME}"
FROZEN_CKPT = FROZEN_DIR / "ppo_candy_model.zip"

# 评估配置
N_EPISODES = 100
BASE_SEED = 20260519
ENV_KWARGS = {
    "board_size": 8,
    "n_colors": 6,
    "max_steps": 100,
    "hole_ratio": 0.08,
    "invalid_penalty": -0.5,
    "no_match_penalty": -0.01,
    "potential_weight": 0.25,
    "move_bonus": 0.1,
}


def freeze_checkpoint() -> None:
    if not SOURCE_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {SOURCE_CKPT}")
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_CKPT, FROZEN_CKPT)

    # 额外复制同run的指标文件方便追溯
    run_dir = LOG_ROOT / SOURCE_RUN_NAME
    for name in ["metrics.csv", "metrics.jsonl", "config_snapshot.yaml"]:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, FROZEN_DIR / name)


def evaluate_offline_100() -> dict:
    model = PPO.load(str(FROZEN_CKPT), device="cpu")

    episode_rewards: list[float] = []
    action_counts = {"illegal": 0, "no_match": 0, "scoring": 0}

    for i in range(N_EPISODES):
        env = CandyCrushEnv(**ENV_KWARGS)
        obs, _ = env.reset(seed=BASE_SEED + i)

        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += float(reward)

            at = info.get("action_type", "no_match")
            if at not in action_counts:
                action_counts[at] = 0
            action_counts[at] += 1

        episode_rewards.append(ep_reward)

    rewards = np.array(episode_rewards, dtype=np.float64)
    total_actions = max(1, sum(action_counts.values()))

    return {
        "source_run_name": SOURCE_RUN_NAME,
        "source_checkpoint": str(SOURCE_CKPT),
        "frozen_checkpoint": str(FROZEN_CKPT),
        "episodes": N_EPISODES,
        "seed_start": BASE_SEED,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "p50_reward": float(np.percentile(rewards, 50)),
        "p90_reward": float(np.percentile(rewards, 90)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "scoring_ratio": float(action_counts.get("scoring", 0) / total_actions),
        "illegal_ratio": float(action_counts.get("illegal", 0) / total_actions),
        "action_counts": action_counts,
        "env_kwargs": ENV_KWARGS,
    }


def main() -> None:
    freeze_checkpoint()
    summary = evaluate_offline_100()

    out_json = ROOT / "offline_eval_100_summary.json"
    out_csv = ROOT / "offline_eval_100_summary.csv"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    import pandas as pd

    pd.DataFrame([summary]).to_csv(out_csv, index=False, encoding="utf-8")

    print("✅ Frozen checkpoint:", FROZEN_CKPT)
    print("✅ Saved:")
    print("-", out_json)
    print("-", out_csv)
    print("\n===== OFFLINE EVAL (100 episodes) =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
