from __future__ import annotations

"""A/B v3 benchmark: Uniform vs harder Beta hole-ratio sampling.

This script is designed for Colab usage and project-local usage.
It trains two 10k continuation runs from the same checkpoint and compares:
- A_uniform: hole_ratio ~ Uniform(0.04, 0.08)
- B_beta_hard: hole_ratio ~ Beta(alpha=5, beta=2) mapped to [0.04, 0.08]

Outputs:
- ab_v3_10k_summary.csv
- ab_v3_10k_summary.json
- ab_v3_10k_report.json
"""

from dataclasses import asdict
from pathlib import Path
import json
import shutil
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from candy_rl.env import CandyCrushEnv
from candy_rl.train_sb3 import (
    ActionMaskWrapper,
    MetricsCallback,
    linear_lr_schedule,
    load_config,
    make_env_kwargs,
    make_run_dir,
)


ROOT = Path("/content/drive/MyDrive/candybot")
CFG_PATH = ROOT / "configs/train.yaml"
BACKUP_ROOT = Path("/content/drive/MyDrive/candy_backups")
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

START_CKPT = (
    Path("/content/drive/MyDrive/candy_backups")
    / "best_of_run_20k_staged_20260519_144304"
    / "ppo_candy_model_best_of_run.zip"
)
PLATFORM = 45.78
TOTAL_STEPS = 10_000
LOW, HIGH = 0.04, 0.08

CANDIDATES = [
    {
        "name": "A_uniform",
        "sampler": "uniform",
        "learning_rate": 0.00030,
        "learning_rate_final": 0.00016,
        "clip_range": 0.24,
        "ent_coef": 0.01,
    },
    {
        "name": "B_beta_hard",
        "sampler": "beta",
        "beta_alpha": 5.0,
        "beta_beta": 2.0,
        "learning_rate": 0.00030,
        "learning_rate_final": 0.00016,
        "clip_range": 0.24,
        "ent_coef": 0.01,
    },
]


class HoleSamplerWrapper(gym.Wrapper):
    """Apply per-episode hole-ratio sampling to underlying CandyCrushEnv."""

    def __init__(
        self,
        env: gym.Env,
        mode: str,
        low: float,
        high: float,
        beta_alpha: float = 5.0,
        beta_beta: float = 2.0,
    ):
        super().__init__(env)
        self.mode = mode
        self.low = float(low)
        self.high = float(high)
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)

    def _find_base_env(self):
        cur = self.env
        while hasattr(cur, "env"):
            if isinstance(cur, CandyCrushEnv):
                return cur
            cur = cur.env
        return cur if isinstance(cur, CandyCrushEnv) else None

    def _sample_ratio(self) -> float:
        if self.mode == "uniform":
            return float(np.random.uniform(self.low, self.high))
        if self.mode == "beta":
            x = float(np.random.beta(self.beta_alpha, self.beta_beta))
            return self.low + x * (self.high - self.low)
        return float(np.random.uniform(self.low, self.high))

    def reset(self, **kwargs):
        base = self._find_base_env()
        if base is not None:
            base.hole_ratio = self._sample_ratio()
        return self.env.reset(**kwargs)


def summarize(run_dir: Path) -> dict:
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return {}
    df = pd.read_csv(metrics)
    out: dict[str, float | int] = {
        "rows_total": int(len(df)),
        "timesteps_max": int(df["timesteps"].max()) if "timesteps" in df.columns and len(df) else 0,
    }

    for c in ["eval_mean_reward", "eval_std_reward", "eval_scoring_ratio", "eval_illegal_ratio"]:
        if c not in df.columns:
            df[c] = np.nan

    ev = df[df["eval_mean_reward"].notna()].copy()
    out["eval_points"] = int(len(ev))
    if len(ev):
        out["eval_best_mean"] = float(ev["eval_mean_reward"].max())
        out["eval_final_mean"] = float(ev["eval_mean_reward"].iloc[-1])
        tail5 = ev.tail(min(5, len(ev)))
        out["last5_eval_mean_avg"] = float(tail5["eval_mean_reward"].mean())
        out["eval_scoring_ratio_last5_avg"] = float(tail5["eval_scoring_ratio"].mean())
        out["eval_illegal_ratio_last5_avg"] = float(tail5["eval_illegal_ratio"].mean())

    return out


def score_fn(row: dict) -> float:
    mean_v = float(row.get("last5_eval_mean_avg", -1e9))
    score_v = float(row.get("eval_scoring_ratio_last5_avg", 0.0))
    illegal_v = float(row.get("eval_illegal_ratio_last5_avg", 1.0))
    penalty = 1000.0 if illegal_v > 0.0 else 0.0
    return mean_v + 2200.0 * score_v - penalty


def run_one(candidate: dict) -> dict:
    cfg = load_config(str(CFG_PATH))
    cfg.total_timesteps = TOTAL_STEPS
    cfg.learning_rate = float(candidate["learning_rate"])
    cfg.learning_rate_final = float(candidate["learning_rate_final"])
    cfg.clip_range = float(candidate["clip_range"])
    if hasattr(cfg, "ent_coef"):
        cfg.ent_coef = float(candidate["ent_coef"])

    env_kwargs = make_env_kwargs(cfg)

    def _make_env():
        env: gym.Env = CandyCrushEnv(**env_kwargs)
        if cfg.use_legal_action_filter:
            env = ActionMaskWrapper(env)
        env = HoleSamplerWrapper(
            env,
            mode=str(candidate["sampler"]),
            low=LOW,
            high=HIGH,
            beta_alpha=float(candidate.get("beta_alpha", 5.0)),
            beta_beta=float(candidate.get("beta_beta", 2.0)),
        )
        return env

    run_dir = make_run_dir(cfg.run_root)
    env = make_vec_env(_make_env, n_envs=cfg.n_envs, seed=cfg.seed)

    model = PPO.load(str(START_CKPT), env=env, device=cfg.device)
    model.lr_schedule = linear_lr_schedule(cfg.learning_rate, cfg.learning_rate_final)
    model.clip_range = lambda _progress: cfg.clip_range
    model.ent_coef = float(candidate["ent_coef"])

    if cfg.use_legal_action_filter:
        for e in env.envs:
            cur = e
            while hasattr(cur, "env"):
                if isinstance(cur, ActionMaskWrapper):
                    cur.model = model
                    break
                cur = cur.env

    cb = MetricsCallback(
        run_dir=Path(run_dir),
        eval_freq=cfg.eval_freq,
        eval_episodes=cfg.eval_episodes,
        eval_deterministic=cfg.eval_deterministic,
        env_kwargs=env_kwargs,
        use_filter=cfg.use_legal_action_filter,
        verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb, progress_bar=False)
    wall = time.time() - t0

    model.save(str(Path(run_dir) / "ppo_candy_model"))
    with open(Path(run_dir) / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)

    summary = {
        "variant": candidate["name"],
        "sampler": candidate["sampler"],
        "run_name": Path(run_dir).name,
        "run_dir": str(run_dir),
        "source_checkpoint": str(START_CKPT),
        "continue_steps": TOTAL_STEPS,
        "learning_rate": candidate["learning_rate"],
        "learning_rate_final": candidate["learning_rate_final"],
        "clip_range": candidate["clip_range"],
        "ent_coef": candidate["ent_coef"],
        "wall_time_sec": round(float(wall), 2),
    }
    if candidate["sampler"] == "beta":
        summary["beta_alpha"] = candidate.get("beta_alpha", 5.0)
        summary["beta_beta"] = candidate.get("beta_beta", 2.0)
    summary.update(summarize(Path(run_dir)))
    summary["break_platform_45_78"] = bool(float(summary.get("last5_eval_mean_avg", -1e9)) > PLATFORM)
    summary["score_for_rank"] = score_fn(summary)
    return summary


def main() -> None:
    rows = []
    for c in CANDIDATES:
        print(f"\n===== Running {c['name']} =====")
        s = run_one(c)
        rows.append(s)
        print(
            {
                "variant": s["variant"],
                "sampler": s["sampler"],
                "run_name": s["run_name"],
                "last5_eval_mean_avg": s.get("last5_eval_mean_avg"),
                "eval_scoring_ratio_last5_avg": s.get("eval_scoring_ratio_last5_avg"),
                "break_platform_45_78": s.get("break_platform_45_78"),
                "score_for_rank": s.get("score_for_rank"),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["break_platform_45_78", "score_for_rank", "last5_eval_mean_avg"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df["rank"] = df.index + 1

    out_csv = ROOT / "ab_v3_10k_summary.csv"
    out_json = ROOT / "ab_v3_10k_summary.json"
    out_report = ROOT / "ab_v3_10k_report.json"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    winner = df.iloc[0].to_dict()
    winner_run = Path(winner["run_dir"])
    winner_backup = BACKUP_ROOT / f"abv3_top1_{winner['variant']}_{winner['run_name']}"
    winner_backup.mkdir(parents=True, exist_ok=True)

    for name in ["ppo_candy_model.zip", "metrics.csv", "metrics.jsonl", "config_snapshot.yaml"]:
        src = winner_run / name
        if src.exists():
            shutil.copy2(src, winner_backup / name)
    if (winner_run / "tb").exists():
        shutil.copytree(winner_run / "tb", winner_backup / "tb", dirs_exist_ok=True)

    report = {
        "platform": PLATFORM,
        "start_checkpoint": str(START_CKPT),
        "winner": winner,
        "winner_backup_dir": str(winner_backup),
        "summary_csv": str(out_csv),
        "summary_json": str(out_json),
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== FINAL RANK =====")
    print(
        df[
            [
                "rank",
                "variant",
                "sampler",
                "run_name",
                "last5_eval_mean_avg",
                "eval_scoring_ratio_last5_avg",
                "eval_illegal_ratio_last5_avg",
                "break_platform_45_78",
                "score_for_rank",
            ]
        ].to_string(index=False)
    )
    print("\n✅ Saved:")
    print("-", out_csv)
    print("-", out_json)
    print("-", out_report)
    print("-", winner_backup)


if __name__ == "__main__":
    main()
