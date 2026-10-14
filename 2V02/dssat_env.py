import gym
from gym import spaces
import numpy as np
import os
import subprocess
from gym_dssat_pdi.envs.utils import utils as dssat_utils

class GymDssatEnv(gym.Env):
    """
    Gym environment for DSSAT-PDI simulations.
    This is a skeleton: step() currently returns dummy data.
    Later, you can replace it with actual DSSAT calls.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, run_dssat_location, log_saving_path='./logs/dssat.log',
                 mode='fertilization', seed=123, random_weather=False):
        super().__init__()

        # -----------------------
        # DSSAT config
        # -----------------------
        self.run_dssat_location = run_dssat_location
        self.log_saving_path = log_saving_path
        self.mode = mode
        self.seed = seed
        self.random_weather = random_weather

        # Create logs folder if it doesn't exist
        dssat_utils.make_folder(os.path.dirname(log_saving_path))

        # -----------------------
        # Gym action & observation spaces
        # -----------------------
        # Example: action = fertilizer amount (0-200 kg/ha)
        self.action_space = spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32)

        # Example: observation = 5 crop / soil / weather features
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)

        # Random state for reproducibility
        self.np_random = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        """
        # Optionally set seed
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # TODO: prepare DSSAT input files here

        # Dummy observation
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs

    def step(self, action):
        """
        Execute one step in the environment.
        Replace dummy simulation with DSSAT call.
        """
        # Clip action to action_space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # -----------------------
        # TODO: Write DSSAT input files here
        # e.g., modify .X files with fertilizer, irrigation, etc.
        # -----------------------

        # -----------------------
        # TODO: Run DSSAT simulation
        # Example (WSL/Linux):
        # subprocess.run([self.run_dssat_location, "/some/path/your.executable"], check=True)
        # -----------------------

        # -----------------------
        # TODO: Parse DSSAT output files to get observation and reward
        # For now, dummy data:
        obs = self.np_random.random(self.observation_space.shape).astype(np.float32)
        reward = float(np.sum(action))  # dummy reward: sum of applied fertilizer
        done = False  # could set True if simulation ends
        info = {}    # optional additional info

        return obs, reward, done, info

    def render(self, mode='human'):
        # Optional: print info to console
        print(f"Step: mode={self.mode}")

    def close(self):
        # Optional: clean up temporary files if needed
        pass
