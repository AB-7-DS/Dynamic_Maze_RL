# Intelligent Maze Solver using PPO and Wider Perception

This project demonstrates how to train a Reinforcement Learning (RL) agent that can intelligently navigate dynamic mazes. The agent is trained using the Proximal Policy Optimization (PPO) algorithm from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

The core challenge addressed is creating an agent that doesn't just memorize a single path but learns a general navigation *policy*. This allows it to dynamically reroute when its preferred path is blocked by obstacles that change in every episode.

## Demo Video

You can view a demonstration of the environment in action by clicking the link below:

[Watch the Maze Demo Video](/demo.mp4)


## The Problem: Myopic Agents

A common issue in simple grid-world environments is that the agent has a very limited view of its surroundings (e.g., only the 4 adjacent cells). This is a **myopic** observation. While the agent can learn an optimal path for a *static* maze, it fails in a *dynamic* one. If a wall suddenly appears on its learned path, the agent only discovers it when it's right next to it, by which time it has no robust strategy to find a detour and often gets stuck.

## The Solution: A Wider Field of View

This project solves the problem by enriching the agent's observation space. Instead of just seeing its immediate neighbors, the agent is given a **wider perception grid** (e.g., a 5x5 grid centered on itself).

This provides two key advantages:
1.  **Anticipation:** The agent can see obstacles from several steps away, giving it time to adjust its trajectory.
2.  **Contextual Awareness:** The agent learns a policy based on the *local layout* of the maze, rather than just its absolute position.

The observation given to the agent includes:
*   **Relative Goal Vector:** A 2D vector pointing from the agent to the goal (`[delta_row, delta_col]`).
*   **Perception Grid:** A flattened array representing the 5x5 area around the agent, where each cell is encoded (e.g., `0` for empty, `1` for wall, `3` for goal).

## 📂 Project Structure

Here is the file structure for this project:

```
├Phase3-Dynamic_Maze/
├ └──maze_class.py       # (The custom Gymnasium environment for the dynamic maze)
├ └── ppo.py              # (The main script for training the PPO agent and running tests)
├── requirements.txt    # (Python dependencies)
└── README.md           # (This file)
```

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>

    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

   

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 How to Run

1.  **Start Training:**
    Simply run the `ppo.py` script. It will automatically start the training process.
    ```bash
    python ppo.py
    ```
    The script will:
    - Instantiate the vectorized and normalized maze environment.
    - Configure and initialize the PPO model.
    - Train the agent for the number of timesteps specified in the script.
    - Save the trained model and the `VecNormalize` statistics to the `models/` directory.

2.  **Watch the Trained Agent:**
    After training is complete, the script automatically loads the saved model and runs 10 test episodes with rendering enabled, so you can visually confirm the agent's performance.

## 🔧 Configuration

You can easily adjust the environment and training parameters at the top of the `ppo.py` file:

*   `NUM_OBSTACLES`: The number of dynamic obstacles placed in the maze each episode.
*   `PERCEPTION_RADIUS`: Controls the size of the agent's field of view. A radius of `2` creates a 5x5 grid.
*   `TOTAL_TIMESTEPS`: The total number of steps to train the agent for. More complex environments require more steps.