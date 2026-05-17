# candy-rl-starter

一个可直接扩展的 RL 工程化模板，覆盖你提到的三项补强目标：

1. **Gymnasium 标准环境接口**（`action_space`, `observation_space`, `reset`, `step`）
2. **训练日志工程化**（`metrics.csv` + `metrics.jsonl` + TensorBoard）
3. **工程化项目结构**（`requirements.txt` / `pyproject.toml` / `configs/*.yaml` / `logging`）

## 快速开始

```bash
pip install -r requirements.txt
bash scripts/train.sh
```

## 训练产物

每次训练会自动生成独立 `run_id` 目录，例如：

- `logs/run_20260517_120000/metrics.csv`
- `logs/run_20260517_120000/metrics.jsonl`
- `logs/run_20260517_120000/tb/` (TensorBoard)
- `logs/run_20260517_120000/ppo_candy_model.zip`
- `logs/run_20260517_120000/config_snapshot.yaml`

## TensorBoard

```bash
tensorboard --logdir logs
```

## 说明

- 这是你当前 Candy 环境的 Gymnasium 规范化版本，便于后续切到 Stable-Baselines3、Gym benchmark、MuJoCo benchmark。
- 下一步建议新增：
  - `train_cartpole.py`（标准 benchmark）
  - `train_hopper.py`（MuJoCo benchmark）
