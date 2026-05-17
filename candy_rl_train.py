import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output


class CandyCrushEnv:
    """Candy Crush 游戏环境（8x8, 6色），支持空位(-1)且空位不可移动。"""

    def __init__(self, board_size=8, n_colors=6, max_steps=100, hole_ratio=0.08):
        self.board_size = board_size
        self.n_colors = n_colors
        self.max_steps = max_steps
        self.hole_ratio = hole_ratio

        self.board = None
        self.score = 0
        self.steps = 0

        # 动作：每个格子向右交换 + 向下交换
        self.n_actions = (board_size * (board_size - 1)) * 2

    def reset(self):
        self.board = np.random.randint(0, self.n_colors, (self.board_size, self.board_size))
        self.score = 0
        self.steps = 0

        # 清掉初始连消
        while self._has_matches():
            self._clear_all_matches()
            self._apply_gravity()
            self._refill()

        # 注入空位，强制训练看到 -1 情况
        self._inject_holes(self.hole_ratio)
        return self._get_observation()

    def _inject_holes(self, hole_ratio=0.08):
        total = self.board_size * self.board_size
        n_holes = int(total * hole_ratio)
        if n_holes <= 0:
            return
        indices = np.random.choice(total, size=n_holes, replace=False)
        rows = indices // self.board_size
        cols = indices % self.board_size
        self.board[rows, cols] = -1

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
            reward = -2  # 非法动作（含空位）惩罚更大

        done = self.steps >= self.max_steps
        return self._get_observation(), reward, done, {'score': self.score}

    def _get_observation(self):
        # 6个颜色通道 + 1个 empty(-1) 通道
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

        # 关键规则：空位不可移动
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

        # 横向
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

        # 纵向
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


class ActorCritic(nn.Module):
    """Actor-Critic 网络，用于 PPO。"""

    def __init__(self, board_size=8, n_channels=7, n_actions=112):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flatten_size = 64 * board_size * board_size

        self.actor = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class PPOTrainer:
    def __init__(self, env, lr=3e-4, gamma=0.99, clip_epsilon=0.2, n_epochs=10, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        test_obs = env.reset()
        self.n_channels = test_obs.shape[0]
        self.board_size = test_obs.shape[1]
        self.n_actions = env.n_actions

        self.model = ActorCritic(self.board_size, self.n_channels, self.n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

        self.episode_rewards = []
        self.episode_lengths = []

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, value = self.model(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def store_experience(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_advantages(self, rewards, values, dones, last_value=0):
        advantages = []
        gae = 0
        lam = 0.95

        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update_policy(self):
        if len(self.states) == 0:
            return

        advantages, returns = self.compute_advantages(self.rewards, self.values, self.dones)

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                action_probs, values = self.model(batch_states)
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

    def train(self, n_episodes=500, update_frequency=2048):
        step_count = 0

        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, log_prob, value = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.store_experience(state, action, reward, done, log_prob, value)
                episode_reward += reward
                episode_length += 1
                state = next_state
                step_count += 1

                if step_count % update_frequency == 0:
                    self.update_policy()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                clear_output(wait=True)
                print(f"Episode {episode}/{n_episodes}")
                print(f"Average Reward (last 10): {avg_reward:.2f}")
                print(f"Best Reward: {max(self.episode_rewards):.2f}")
                self._plot_progress()

    def _plot_progress(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.subplot(1, 2, 2)
        if len(self.episode_rewards) >= 10:
            smoothed = np.convolve(self.episode_rewards, np.ones(10) / 10, mode='valid')
            plt.plot(smoothed)
            plt.title('Smoothed Rewards (window=10)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')

        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


def test_model(env, trainer, n_episodes=5):
    print("\n" + "=" * 50)
    print("Testing Model...")
    print("=" * 50)

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action, _, _ = trainer.get_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

    print("=" * 50)


if __name__ == "__main__":
    print("=" * 50)
    print("Candy Crush PPO Training (with empty-cell rule)")
    print("=" * 50)

    env = CandyCrushEnv(board_size=8, n_colors=6, max_steps=100, hole_ratio=0.08)
    print(f"Environment created: {env.board_size}x{env.board_size} board")
    print(f"Action space: {env.n_actions} actions")
    print("Rule enabled: empty cell (-1) cannot be swapped")

    trainer = PPOTrainer(env=env, lr=3e-4, gamma=0.99, clip_epsilon=0.2, n_epochs=10, batch_size=64)

    print("\nStarting training...")
    trainer.train(n_episodes=500, update_frequency=2048)

    trainer.save_model("/content/candy_crush_model.pth")
    test_model(env, trainer, n_episodes=3)

    print("\n✅ Training complete!")
    print("Model saved to /content/candy_crush_model.pth")
