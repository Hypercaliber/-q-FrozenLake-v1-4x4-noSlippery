# Q-Learning on FrozenLake-v1 (4x4, non-slippery)

A tabular Q-learning agent that learns to solve the 4x4 FrozenLake environment from [Gymnasium](https://gymnasium.farama.org/). I built it while working through Unit 2 of the [Hugging Face Deep Reinforcement Learning course](https://huggingface.co/learn/deep-rl-course).

**Result:** mean reward of **1.00 ± 0.00** over the evaluation episodes, meaning the agent reaches the goal every time.

## The Environment

FrozenLake is a grid world. The agent starts in the top-left corner and must reach the goal in the bottom-right without falling into a hole.

- **States:** 16 (one per cell, numbered 0 to 15)
- **Actions:** 4 (left, down, right, up)
- **Reward:** 1 for reaching the goal, 0 otherwise
- **Slippery:** off (`is_slippery=False`), so every move goes exactly where intended

## How It Works

The agent keeps a Q-table with one row per state and one column per action. Each entry estimates how good an action is in a given state, and it is updated after every step with the Q-learning (temporal difference) rule:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max Q(s', a') − Q(s, a) ]
```

During training it uses an epsilon-greedy policy, exploring randomly at first and exploiting its learned values more as epsilon decays. At evaluation time it always picks the action with the highest Q-value.

## Using the Trained Model

The trained Q-table is stored in `q-learning.pkl`. To load it from the Hugging Face Hub:

```python
import pickle
import gymnasium as gym
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="BiggestQ/q-FrozenLake-v1-4x4-noSlippery",
    filename="q-learning.pkl",
)
with open(path, "rb") as f:
    model = pickle.load(f)

# The pickle also stores the environment settings (e.g. is_slippery=False)
env = gym.make(model["env_id"], is_slippery=False)
```

## Video Replay


https://github.com/user-attachments/assets/975235a5-be33-4e16-8d9b-ff17b359338f



## Links

- Model on Hugging Face: https://huggingface.co/BiggestQ/q-FrozenLake-v1-4x4-noSlippery
- Course: https://huggingface.co/learn/deep-rl-course
