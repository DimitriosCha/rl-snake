🐍 Reinforcement-Learning Snake

A modular, RL-ready Snake environment with DQN training and agent visualization

This project implements a full reinforcement-learning environment for the classic Snake game — complete with a Gym-like API, modular game logic, multiple policies (random, greedy, ε-greedy), and a full Deep Q-Network (DQN) training pipeline.

The goal of the project is learning by building:
from raw game mechanics → to environment design → to training deep RL agents → to live visualization of the trained agent playing in a pygame window.

----------------------------------------------------------------------------------------

<img width="543" height="514" alt="Screenshot 2025-11-13 at 11 15 04 pm" src="https://github.com/user-attachments/assets/59dc0974-21a6-40f6-b57e-463204f85add" />


----------------------------------------------------------------------------------------

🎯 Observation Space

Each state is a 9-dimensional vector:

| Index | Feature        | Meaning                         |
| ----- | -------------- | ------------------------------- |
| 0     | `hx_n`         | head x (normalized 0–1)         |
| 1     | `hy_n`         | head y (normalized 0–1)         |
| 2     | `fx_n`         | food x (normalized 0–1)         |
| 3     | `fy_n`         | food y (normalized 0–1)         |
| 4     | `dx`           | direction x (−1, 0, 1)          |
| 5     | `dy`           | direction y (−1, 0, 1)          |
| 6     | `danger_ahead` | 1 if next move forward is fatal |
| 7     | `danger_left`  | 1 if left turn is fatal         |
| 8     | `danger_right` | 1 if right turn is fatal        |

----------------------------------------------------------------------------------------

🏆 Reward Function
| Event               | Reward                         |
| ------------------- | ------------------------------ |
| Eat food            | `+1.0`                         |
| Die                 | `-1.0`                         |
| Step penalty        | `-0.001`                       |
| Move closer to food | `+0.01 * (d_before - d_after)` |


Reward shaping encourages exploration and reduces wandering, while still letting the agent learn strategic behavior.

----------------------------------------------------------------------------------------

🚀 Training

Train a DQN agent:

python -m src.rl.train --policy dqn --episodes 3000

This will produce:
* CSV logs under data/runs/rl_dqn.csv
* A saved model under data/runs/rl_dqn_dqn.pt



----------------------------------------------------------------------------------------

📊 Future Improvements

Environment Improvements:
*   Add a “local grid” observation (5×5 or 7×7 vision)
*   RL Improvements
*   Double DQN  D
*   Dueling DQN
*   Policy-gradient agents (PPO, A2C)
*   TensorBoard logging
*   Live performance dashboard


📝 License
MIT — free to use, modify, and distribute.

🤝 Contributing -- Contributions are welcome — especially around:

* improving the agent architecture
* extending the observation space
* adding alternative RL algorithms
