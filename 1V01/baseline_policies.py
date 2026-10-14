# Baseline agents for comparison
from sb3_wrapper import Formator
import numpy as np

class NullAgent:
    """Agent always choosing to do no fertilization"""
    def __init__(self, env):
        self.env = env
        # 兼容不同层级：Monitor→GymDssatWrapper→DssatPdi 或 GymDssatWrapper→DssatPdi
        self.action_formator = Formator(self._get_dssat_env(env))

    def _get_dssat_env(self, env):
        """一直剥到 DssatPdi 层"""
        inner = env
        while hasattr(inner, 'env'):
            inner = inner.env
        return inner

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        normalized_action = self.action_formator.normalize_actions([0])
        return np.array(normalized_action, dtype=np.float32), obs


class ExpertAgent:
    """Simple agent using policy based on days after planting"""
    def __init__(self, env):
        self.env = env
        dssat_env = self._get_dssat_env(env)
        self.action_formator = Formator(dssat_env)

        # 从任意层级拿 observation_variables
        obs_vars = self._get_obs_vars(env)
        assert 'dap' in obs_vars, f"'dap' not found in observation_variables: {obs_vars}"
        self.dap_index = obs_vars.index('dap')

        # mode 也从底层拿
        self.mode = dssat_env.mode

        all_policy_dic = {
            'fertilization': {
                1: 165 # 138 + 27
                # 海伦站
                # 41:207
                # 禹城站
                # 30: 40.5
                # 封丘站
            },
            'irrigation': {
                # 1: 75,
                # 封丘站
                # 1: 120,
                # 禹城站
                49: 10,
                70: 10,
                95: 10
                # 海伦站
                # 5: 70,
                # 43: 60
                # 栾城站、沈阳站
            }
        }
        self.policy_dic = all_policy_dic[self.mode]

    def _get_dssat_env(self, env):
        inner = env
        while hasattr(inner, 'env'):
            inner = inner.env
        return inner

    def _get_obs_vars(self, env):
        """从任意层级的 env 里拿 observation_variables"""
        inner = env
        while inner is not None:
            if hasattr(inner, 'observation_variables'):
                return inner.observation_variables
            inner = getattr(inner, 'env', None)
        raise AttributeError("找不到 observation_variables，检查 env 层级")

    def _policy(self, obs):
        obs = np.concatenate(obs, axis=None)
        dap = int(obs[self.dap_index])
        action = [self.policy_dic[dap] if dap in self.policy_dic else 0]
        return action

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        action = self._policy(obs)
        normalized_action = self.action_formator.normalize_actions(action)
        return np.array(normalized_action, dtype=np.float32), obs
