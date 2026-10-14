# import gym
import numpy as np
import pdb
import gymnasium as gym


class Formator():
    def __init__(self, env):
        self.action_space_dict = env.action_space
        self.action_names = [*env.action_space]
        self.observation_dict_to_array = env.observation_dict_to_array

    def _get_action_bounds(self, action_name):
        return (self.action_space_dict[action_name].low, self.action_space_dict[action_name].high)

    def _check_array_actions(self, actions):
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions]
        return actions

    # helpers for action normalization
    def normalize_actions(self, actions):
        """Normalize the action from [low, high] to [-1, 1]"""
        actions = self._check_array_actions(actions)
        normalized_actions = []
        for action_name, action in zip(self.action_names, actions):
            low, high = self._get_action_bounds(action_name)
            normalized_action = 2.0 * ((action - low) / (high - low)) - 1.0
            normalized_actions.append(normalized_action)
        return normalized_actions

    def denormalize_actions(self, actions):
        """Denormalize the action from [-1, 1] to [low, high]"""
        actions = self._check_array_actions(actions)
        denormalized_actions = []
        for action_name, action in zip(self.action_names, actions):
            low, high = self._get_action_bounds(action_name)
            denormalized_action = low + (0.5 * (action + 1.0) * (high - low))
            denormalized_actions.append(denormalized_action)
        return denormalized_actions

    def format_actions(self, actions):
        actions = self._check_array_actions(actions)
        return {action_name: action for action_name, action in zip([*self.action_space_dict], actions)}

    def format_observation(self, observation):
        return self.observation_dict_to_array(observation)

class FixInfoWrapper:
    """拦截 gym_dssat_pdi 返回 info=None 的问题"""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = getattr(env, 'reward_range', (-float('inf'), float('inf')))
        self.metadata = getattr(env, 'metadata', {})
        # 透传 dssat 特有属性
        self.observation = env.observation
        self.action_names = getattr(env, 'action_names', None)
        self.observation_dict_to_array = env.observation_dict_to_array

    def step(self, action):
        result = self.env.step(action)
        if result is None:
            return None
        obs, reward, done, info = result
        if info is None:
            info = {}
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return [seed]
    
    def close(self):
        return self.env.close()

    def set_evaluation(self):
        return self.env.set_evaluation()

class GymDssatWrapper(gym.Wrapper):
    def __init__(self, env):
        import gym as old_gym
        # 1. 先保存原始底层 env 的引用（剥 Wrapper 之前）
        base = env
        while isinstance(base, old_gym.Wrapper):
            base = base.env
        self._base_env = base          # ← 保存底层引用

        # 2. 修复 info=None
        env = FixInfoWrapper(env)

        # 3. shimmy 转换
        from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
        env = GymV21CompatibilityV0(env=env)

        self.formator = Formator(env)
        super().__init__(env)
        self.env = env

        self.action_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(len(self.formator.action_names),),
            dtype="float32"
        )
        obs_shape = self.formator.format_observation(env.observation).shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf,
            shape=obs_shape,
            dtype="float32"
        )

        self.last_info = {}
        self.last_obs = None
        self.last_obs_dict = None

    # ── 新增：透传底层 history / observation_variables ──────────────
    @property
    def history(self):
        return getattr(self._base_env, 'history', {})

    @property
    def observation_variables(self):
        return getattr(self._base_env, 'observation_variables', [])
    # ────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        # （不变）
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            raw_obs, info = result
        else:
            raw_obs, info = result, {}
        formatted_obs = self.formator.format_observation(raw_obs)

        # 保存原始 dict
        self.last_obs_dict = raw_obs

        self.last_obs = formatted_obs
        self.last_info = info

        return formatted_obs, info

    def step(self, action):
        denormalized_action = self.formator.denormalize_actions(action)
        formatted_action = self.formator.format_actions(denormalized_action)

        result = self.env.step(formatted_action)
        # print("\n========= RAW STEP RESULT =========")
        # print(type(result))
        # print(result)
        # print("===================================\n")

        if result is None or result[0] is None:
            formatted_observation = self.formator.format_observation(self.last_obs) \
                if isinstance(self.last_obs, dict) else self.last_obs
            return formatted_observation, 0.0, True, False, self.last_info

        # shimmy 转换后是5元组
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False

        # 保存原始 observation dict
        self.last_obs_dict = obs

        self.last_obs = obs
        self.last_info = info if info is not None else {}

        formatted_observation = self.formator.format_observation(obs)

        if reward is None:
            reward = 0.0

        return formatted_observation, float(reward), terminated, truncated, self.last_info

    def close(self):
        return self.env.close()

    def eval(self):
        return self.env.set_evaluation()

    def __del__(self):
        self.close()


if __name__ == '__main__':
    pass