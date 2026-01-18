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


class GymDssatWrapper(gym.Wrapper):
    def __init__(self, env):
        self.formator = Formator(env)
        super().__init__(env)
        self.env = env

        self.action_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(len(self.formator.action_names),),
            dtype="float32"
        )

        obs_shape = self.formator.format_observation(
            env.observation
        ).shape

        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf,
            shape=obs_shape,
            dtype="float32"
        )

        self.last_info = {}
        self.last_obs = None

    def reset(self, *, seed=None, options=None):
        raw_obs = self.env.reset()
        formatted_obs = self.formator.format_observation(raw_obs)
        info = {}
        self.last_obs = formatted_obs
        self.last_info = info
        return formatted_obs, info

    def step(self, action):
        denormalized_action = self.formator.denormalize_actions(action)
        formatted_action = self.formator.format_actions(denormalized_action)

        obs, reward, done, info = self.env.step(formatted_action)

        # 這裡是關鍵：你的原始 env 很可能用舊的 "done" 邏輯
        # 我們假設沒有明確的 truncated 情境（大多數自訂 env 都是這樣）
        terminated = done
        truncated = False   # 如果你有 TimeLimit 或 max steps 邏輯，可以設為 True

        if done:
            # 你的原有邏輯：episode 結束時用 last 的值避免 None
            obs, reward, info = self.last_obs, 0.0, self.last_info
        else:
            self.last_obs = obs
            self.last_info = info

        formatted_observation = self.formator.format_observation(obs)

        return formatted_observation, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def eval(self):
        return self.env.set_evaluation()

    def __del__(self):
        self.close()


if __name__ == '__main__':
    pass
