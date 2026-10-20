# Continuous Maze — Deep Q-Network (DQN)

A reinforcement learning agent that navigates a continuous-coordinate maze full of
danger zones, trained with a **Deep Q-Network (DQN)**. Built with a custom
[Gymnasium](https://gymnasium.farama.org/) environment and rendered live with pygame.

## What this project is about

[Project A](../Project_A) in this series uses tabular Q-learning, which only works
because that grid has a small, finite number of states — you can literally store a
value for every (state, action) pair in a table. That approach breaks down here: the
agent's position is a **continuous** `(x, y)` coordinate, so there are infinitely many
possible states, and no table could ever cover them all.

A DQN solves this by replacing the table with a neural network: instead of looking up
a Q-value, the network *estimates* it from the raw position. This lets the agent
generalise — it can make a sensible decision even from a position it has never
visited in exactly that spot before, as long as it's seen *similar* positions during
training.

## The environment (`env.py`)

- **State**: agent position `(x, y)`, continuous, normalized to `[0, 1]²`
- **Actions**: 4 discrete moves — `0=Up`, `1=Down`, `2=Left`, `3=Right`, step size `0.05`
- **Goal**: reach `(0.9, 0.5)` within radius `0.05` → **+10** reward, episode ends
- **Danger zones**: 7 rectangular regions — stepping into one gives **-100** reward and
  ends the episode immediately (a single mistake is costly, so the agent has to learn
  to route around them rather than cut corners)
- **Walls**: 2 rectangular barriers — colliding gives **-1** reward and the agent simply
  stays in place (penalised, but not fatal)
- Rendered live with a pygame window: red = danger zone, black = wall, green = goal,
  blue = agent

## Files

| File | Purpose |
|---|---|
| `env.py` | The Gymnasium environment (`ContinuousMazeEnv`) |
| `DQN_model.py` | The Q-network architecture (`Qnet`) — maps a position to Q-values for each of the 4 actions |
| `utils.py` | Replay buffer (stores past transitions) and the DQN gradient-update step |
| `main.py` | Entry point — train a new agent or test the saved one |
| `dqn_1.pth` | Submitted trained model weights |

## Running it

```bash
pip install -r requirements.txt
python main.py
```

`main.py` is controlled by boolean flags near the top of the file:

| Flag | What it does |
|---|---|
| `test_dqn = True` (default) | Loads `dqn_1.pth`, runs the agent greedily (no exploration), renders the pygame window |
| `train_dqn = True` | Trains a fresh DQN from scratch, saves `dqn_1.pth` + `training_curve.png` |
| `render` | Show the pygame window while testing |
| `save_video` | Save the test rollout to `solution.mp4` |

## How training works

- A **replay buffer** stores past `(state, action, reward, next_state, done)` transitions;
  the network trains on random batches from this buffer rather than only the most recent
  step, which stabilises learning
- A separate **target network** is synced from the main network every 20 episodes — this
  stops the agent from "chasing a moving target" during training
- **ε-greedy exploration**: the agent acts randomly early on (ε starts at 1.0) and
  gradually shifts to acting on its own estimates, annealing to ε=0.01 by the halfway
  point of training
- **Reward shaping** (training only): on top of the real environment reward, the agent
  gets a small bonus proportional to how much closer it moved to the goal this step.
  Without this, the only non-zero rewards are the rare "+10 reached goal" and
  "-100 hit danger zone" — too sparse for the network to learn from efficiently early on.
  The *plotted and reported* reward is always the true, unshaped environment reward —
  shaping only influences what the network trains on, not what's measured as success
- Training stops early once the goal is reached in 20 consecutive episodes (treated as
  "solved")

```
learning_rate = 0.005
gamma         = 0.98
buffer_limit  = 50,000
batch_size    = 32
max_steps     = 300
```

## Results

The submitted `dqn_1.pth` reliably routes around the walls and all 7 danger zones to
reach the goal region. Run `main.py` with the default flags (`test_dqn=True`) to watch
it play.
