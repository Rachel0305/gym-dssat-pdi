# Baseline agents for comparison
from sb3_wrapper import normalize_action
import numpy as np

class NullAgent:
    """
    Agent always choosing to do no fertilization
    """
    def __init__(self, env):
        self.env = env

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        action = normalize_action((self.env.action_low, self.env.action_high), [0])
        return np.array([action], dtype=np.float32), obs


class ExpertAgent:
    """
    Simple agent using policy of choosing fertilization amount based on days after planting
    """
    fertilization_dic = {
        40: 27,
        45: 35,
        80: 54,
    }

    def __init__(self, env, normalize_action=False, fertilization_dic=None):
        self.env = env
        self.normalize_action = normalize_action

    def _policy(self, obs):
        dap = int(obs[0][0])
        return [self.fertilization_dic[dap] if dap in self.fertilization_dic else 0]

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        action = self._policy(obs)
        action = normalize_action((self.env.action_low, self.env.action_high), action)

        return np.array([action], dtype=np.float32), obs