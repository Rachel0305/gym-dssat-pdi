import numpy as np
import gymnasium as gym


class Formator:
    def __init__(self, env):
        self.action_space_dict = env.action_space
        self.action_names = list(env.action_space.keys()) if isinstance(env.action_space, dict) else [*env.action_space]
        self.observation_dict_to_array = getattr(env, 'observation_dict_to_array', lambda x: x)

    def _get_action_bounds(self, action_name):
        space = self.action_space_dict[action_name]
        return (float(space.low), float(space.high))

    def _check_array_actions(self, actions):
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions]
        return actions

    def normalize_actions(self, actions):
        actions = self._check_array_actions(actions)
        normalized = []
        for name, act in zip(self.action_names, actions):
            low, high = self._get_action_bounds(name)
            normalized.append(2.0 * ((act - low) / (high - low)) - 1.0)
        return normalized

    def denormalize_actions(self, actions):
        actions = self._check_array_actions(actions)
        denormalized = []
        for name, act in zip(self.action_names, actions):
            low, high = self._get_action_bounds(name)
            denormalized.append(low + 0.5 * (act + 1.0) * (high - low))
        return denormalized

    def format_actions(self, actions):
        actions = self._check_array_actions(actions)
        return {name: act for name, act in zip(self.action_names, actions)}

    def format_observation(self, observation):
        return self.observation_dict_to_array(observation)


class GymDssatWrapper(gym.Env):
    """
    继承 gymnasium.Env，让 SB3 的 Monitor 能接受。
    内部持有旧版 gym 的 DssatPdi 实例。
    """

    metadata = {'render_modes': []}

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.formator = Formator(env)

        # action_space：连续 [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self.formator.action_names),),
            dtype=np.float32
        )

        # observation_space：先 reset 拿一个样本来确定 shape
        raw_obs = self.env.reset()
        obs_array = np.asarray(self.formator.format_observation(raw_obs), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf,
            shape=obs_array.shape,
            dtype=np.float32
        )

        self._last_obs  = obs_array
        self._last_info = {}

    # ── gymnasium 必须实现的接口 ──────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        raw_obs = self.env.reset()
        obs = np.asarray(
            self.formator.format_observation(raw_obs), dtype=np.float32
        )
        self._last_obs  = obs
        self._last_info = {}
        return obs, self._last_info

    def step(self, action):
        denormalized    = self.formator.denormalize_actions(action)
        formatted_action = self.formator.format_actions(denormalized)

        result = self.env.step(formatted_action)

        # 兼容旧版 gym 的 4-tuple 返回
        if result is None or (isinstance(result, tuple) and result[0] is None):
            return self._last_obs, 0.0, True, False, self._last_info

        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result

        obs = np.asarray(
            self.formator.format_observation(obs), dtype=np.float32
        )
        self._last_obs  = obs
        self._last_info = info if info is not None else {}
        reward = float(reward) if reward is not None else 0.0

        return obs, reward, bool(done), bool(truncated), self._last_info

    def close(self):
        return self.env.close()

    def render(self):
        pass

    # ── 透传属性到底层环境 ─────────────────────────────

    @property
    def unwrapped(self):
        """一直剥到最底层的 DssatPdi"""
        inner = self.env
        while hasattr(inner, 'env'):
            inner = inner.env
        return inner

    def __getattr__(self, name):
        return getattr(self.env, name)

# grok写的
# import numpy as np
# import gymnasium as gym
# from gymnasium.core import Wrapper


# class Formator:
#     def __init__(self, env):
#         self.action_space_dict = env.action_space
#         self.action_names = list(env.action_space.keys()) if isinstance(env.action_space, dict) else [*env.action_space]
#         self.observation_dict_to_array = getattr(env, 'observation_dict_to_array', lambda x: x)

#     def _get_action_bounds(self, action_name):
#         space = self.action_space_dict[action_name]
#         return (float(space.low), float(space.high))

#     def _check_array_actions(self, actions):
#         if not isinstance(actions, (list, np.ndarray)):
#             actions = [actions]
#         return actions

#     def normalize_actions(self, actions):
#         actions = self._check_array_actions(actions)
#         normalized = []
#         for name, act in zip(self.action_names, actions):
#             low, high = self._get_action_bounds(name)
#             normalized.append(2.0 * ((act - low) / (high - low)) - 1.0)
#         return normalized

#     def denormalize_actions(self, actions):
#         actions = self._check_array_actions(actions)
#         denormalized = []
#         for name, act in zip(self.action_names, actions):
#             low, high = self._get_action_bounds(name)
#             denormalized.append(low + 0.5 * (act + 1.0) * (high - low))
#         return denormalized

#     def format_actions(self, actions):
#         actions = self._check_array_actions(actions)
#         return {name: act for name, act in zip(self.action_names, actions)}

#     def format_observation(self, observation):
#         return self.observation_dict_to_array(observation)


# class GymDssatWrapper:
#     """不继承 Wrapper，避免 gymnasium 严格检查"""
    
#     def __init__(self, env):
#         self.env = env
#         self.formator = Formator(env)
        
#         # 定义我们自己的 spaces（给 SB3 使用）
#         self.action_space = gym.spaces.Box(
#             low=-1, high=1,
#             shape=(len(self.formator.action_names),),
#             dtype=np.float32
#         )

#         obs_example = self.formator.format_observation(env.observation)
#         obs_array = np.asarray(obs_example)
#         self.observation_space = gym.spaces.Box(
#             low=0.0, high=np.inf,
#             shape=obs_array.shape,
#             dtype=np.float32
#         )

#         self.last_info = {}
#         self.last_obs = None

#     def reset(self, *, seed=None, options=None):
#         raw_obs = self.env.reset()
#         formatted_obs = self.formator.format_observation(raw_obs)
#         self.last_obs = formatted_obs
#         self.last_info = {}
#         return formatted_obs, self.last_info

#     def step(self, action):
#         denormalized = self.formator.denormalize_actions(action)
#         formatted_action = self.formator.format_actions(denormalized)

#         result = self.env.step(formatted_action)
        
#         if result is None or (isinstance(result, tuple) and result[0] is None):
#             return self.last_obs, 0.0, True, False, self.last_info

#         obs, reward, done, info = result if len(result) == 4 else (result[0], result[1], result[2], {})
        
#         self.last_obs = self.formator.format_observation(obs)
#         self.last_info = info if info is not None else {}
        
#         if reward is None:
#             reward = 0.0

#         return self.last_obs, float(reward), bool(done), False, self.last_info

#     def close(self):
#         return self.env.close()

#     def eval(self):
#         if hasattr(self.env, 'set_evaluation'):
#             return self.env.set_evaluation()

#     def __getattr__(self, name):
#         """转发未定义的方法到原环境"""
#         return getattr(self.env, name)