"""Trusted local recovery bundle, saved at an empty PPO rollout boundary."""
import os,pickle,random
from pathlib import Path

def save_bundle(folder,model,env):
    import numpy as np,torch
    folder=Path(folder);folder.mkdir(exist_ok=False)
    model.save(str(folder/'model.zip'))
    value=dict(step=model.num_timesteps,year=env.current_year,actions=env.replay_actions,
               switch_log=env.switch_log,reset_counts=env.reset_counts,year_rng=env.rng.bit_generator.state,
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
               last_obs=model._last_obs,last_episode_starts=model._last_episode_starts,
               monitor={k:getattr(model.get_env().envs[0],k) for k in ['needs_reset','rewards','total_steps','episode_returns','episode_lengths','episode_times','current_reset_info']},episode_num=model._episode_num,ep_info_buffer=model.ep_info_buffer,ep_success_buffer=model.ep_success_buffer)
    with (folder/'state.pkl').open('xb') as f:pickle.dump(value,f);f.flush();os.fsync(f.fileno())
    (folder/'COMPLETE').write_text(str(value['step']))

def restore_bundle(folder,env):
    import numpy as np,torch
    from sb3_contrib import MaskablePPO
    folder=Path(folder);assert (folder/'COMPLETE').exists()
    with (folder/'state.pkl').open('rb') as f:v=pickle.load(f)
    env.current_year=v['year']
    live=env.current_env;obs,info=live.reset()
    for action in v['actions']:
        obs,reward,term,trunc,info=live.step(action)
        assert not term and not trunc,'Replay crossed saved current episode boundary'
    assert np.array_equal(np.asarray(obs),np.asarray(v['last_obs'])[0]),'Reconstructed DSSAT observation mismatch'
    env.switch_log=v['switch_log'];env.reset_counts=v['reset_counts'];env.rng.bit_generator.state=v['year_rng'];env.replay_actions=list(v['actions'])
    model=MaskablePPO.load(str(folder/'model.zip'),env=env,device='cpu',force_reset=False)
    model._last_obs=v['last_obs'];model._last_episode_starts=v['last_episode_starts']
    for k,value in v['monitor'].items():setattr(model.get_env().envs[0],k,value)
    model._episode_num=v['episode_num'];model.ep_info_buffer=v['ep_info_buffer'];model.ep_success_buffer=v['ep_success_buffer']
    random.setstate(v['python_rng']);np.random.set_state(v['numpy_rng']);torch.set_rng_state(v['torch_rng'])
    return model
