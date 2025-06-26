# train_ppo_wide.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_checker import check_env
import pygame
import os

# Assuming your new environment class is saved in 'wide_perception_maze.py'
from f import MazeGameEnv 

base_maze_config = [
    ['S', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', 'D', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', 'D', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', 'D', '.', '.', '.', '.', '.', 'G', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
]

### --- KEY PARAMETERS --- ###
# We can now be more ambitious with the number of obstacles
NUM_OBSTACLES = 15 
# A 5x5 grid (radius=2) is a great starting point for this maze size.
PERCEPTION_RADIUS = 2 
# The agent needs more time to learn from the richer observation space!
TOTAL_TIMESTEPS = 750000 
MODEL_DIR = "models"
LOG_DIR = "logs"
MODEL_NAME = f"ppo_maze_r{PERCEPTION_RADIUS}_{NUM_OBSTACLES}obs"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    def env_creator(render_mode_val=None):
        return MazeGameEnv(
            base_maze=base_maze_config,
            num_obstacles=NUM_OBSTACLES,
            perception_radius=PERCEPTION_RADIUS, # Pass the new parameter
            render_mode=render_mode_val
        )

    print("Checking the 'Wider Perception' environment...")
    check_env(env_creator())
    print("Environment check passed!")

    # Vectorized environment for training
    vec_env = make_vec_env(lambda: env_creator(), n_envs=8) # More envs can speed up training
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, gamma=0.99)

    # Use a larger network for the more complex observation space
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, MODEL_NAME + "_tb"),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4, # Default is often fine
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=policy_kwargs # Use the larger network
    )

    print(f"Training PPO agent with {PERCEPTION_RADIUS*2+1}x{PERCEPTION_RADIUS*2+1} Perception for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save("Phase3")
    print("Training finished.")

    MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.zip")
    VEC_NORM_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_vecnormalize.pkl")
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    vec_env.save(VEC_NORM_PATH)
    
    # --- Testing Loop ---
    print("\n--- Testing the trained agent ---")
    
    loaded_model = PPO.load(MODEL_PATH)
    
    # Create the test environment
    test_env = env_creator(render_mode_val='human')
    # Important: We need a VecNormalize wrapper for the test env as well
    test_vec_env = VecNormalize.load(VEC_NORM_PATH, make_vec_env(lambda: test_env, n_envs=1))
    
    test_vec_env.training = False
    test_vec_env.norm_reward = False

    for episode in range(10):
        obs = test_vec_env.reset()
        terminated = False
        total_episode_reward = 0
        current_steps = 0
        print(f"\n--- Starting Test Episode {episode + 1} ---")
        
        # Give us time to see the starting board
        test_vec_env.render()
        pygame.time.wait(1500) 

        while not terminated:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, info = test_vec_env.step(action)
            
            # Since test_vec_env is a vectorized env (even with n_envs=1), we access reward[0]
            total_episode_reward += reward[0] 
            
            test_vec_env.render()
            pygame.time.wait(100) # Slow down rendering for visualization
            current_steps += 1
            if current_steps > 200: # Add a max step limit to prevent infinite loops
                print("FAILURE: Episode timed out.")
                break
        
        # Check the 'real' termination condition from the info dict
        if info[0].get('TimeLimit.truncated', False):
            print(f"FAILURE: Episode was truncated (timed out). Reward: {total_episode_reward:.2f}")
        elif total_episode_reward > 5: # Success based on reaching the goal
             print(f"SUCCESS: Episode solved in {current_steps} steps! Reward: {total_episode_reward:.2f}")
        else: # Failure based on hitting a 'D' cell
             print(f"FAILURE: Agent hit a danger zone. Reward: {total_episode_reward:.2f}")

    test_vec_env.close()

if __name__ == '__main__':
    main()