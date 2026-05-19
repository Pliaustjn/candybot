from __future__ import annotations

"""Dual-track offline evaluation for one checkpoint.

Track A: raw environment (no action mask wrapper)
Track B: environment wrapped with ActionMaskWrapper

Purpose:
- diagnose whether performance bottleneck is policy itself or reliance on wrapper filtering.

Usage (Colab):
  PYTHONPATH=/content/drive/MyDrive/candybot/src \
  python scripts/offline_eval_100_dual_track.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from candy_rl.env import CandyCrushEnv
from candy_rl.train_sb3 import ActionMaskWrapper


ROOT = Path("/content/drive/MyDrive/candybot")
LOG_ROOT = ROOT / "logs"

SOURCE_RUN_NAME = "run_20260519_132905"
CHECKPOINT = LOG_ROOT / SOURCE_RUN_NAME / "ppo_candy_model.zip"

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


def run_track(model: PPO, use_mask_wrapper: bool) -> dict:
    episode_rewards: list[float] = []
    action_counts = {"illegal": 0, "no_match": 0, "scoring": 0}

    for i in range(N_EPISODES):
        env = CandyCrushEnv(**ENV_KWARGS)
        if use_mask_wrapper:
            wrapped = ActionMaskWrapper(env)
            wrapped.model = model
            env = wrapped

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
    }


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model = PPO.load(str(CHECKPOINT), device="cpu")

    raw_summary = run_track(model, use_mask_wrapper=False)
    mask_summary = run_track(model, use_mask_wrapper=True)

    result = {
        "source_run_name": SOURCE_RUN_NAME,
        "checkpoint": str(CHECKPOINT),
        "env_kwargs": ENV_KWARGS,
        "track_raw_env": raw_summary,
        "track_mask_wrapped": mask_summary,
        "delta_mean_reward_mask_minus_raw": mask_summary["mean_reward"] - raw_summary["mean_reward"],
        "delta_illegal_ratio_mask_minus_raw": mask_summary["illegal_ratio"] - raw_summary["illegal_ratio"],
        "delta_scoring_ratio_mask_minus_raw": mask_summary["scoring_ratio"] - raw_summary["scoring_ratio"],
    }

    out_json = ROOT / "offline_eval_100_dual_track_summary.json"
    out_csv = ROOT / "offline_eval_100_dual_track_summary.csv"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        {"track": "raw_env", **raw_summary},
        {"track": "mask_wrapped", **mask_summary},
    ]
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    print("✅ Saved:")
    print("-", out_json)
    print("-", out_csv)
    print("\n===== DUAL TRACK OFFLINE EVAL (100 episodes) =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
