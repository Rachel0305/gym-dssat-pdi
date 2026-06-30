
import json, sys, importlib.util
from pathlib import Path
import gym
PROJECT_ROOT=Path('/workspace')
sys.path.insert(0,str(PROJECT_ROOT)); sys.path.insert(0,str(PROJECT_ROOT/'src'))
# install official reward scalarized
spec=importlib.util.spec_from_file_location('gym_dssat_pdi.envs.configs.rewards', PROJECT_ROOT/'references'/'rewards.py')
mod=importlib.util.module_from_spec(spec); sys.modules['gym_dssat_pdi.envs.configs.rewards']=mod; spec.loader.exec_module(mod)
orig=mod.all_reward
import numpy as np
def scalar_all(*a,**k):
    v=orig(*a,**k)
    return float(np.nansum(np.asarray(v,dtype=float))) if isinstance(v,(list,tuple,np.ndarray)) else v
mod.all_reward=scalar_all
mod.get_reward_function=lambda mode: scalar_all if mode=='all' else mod.get_reward_function(mode)
case=PROJECT_ROOT/'DSSAT_auto_validation'/'HLA_2004'/'hla2010_2015_official_reward_restart'/'action_channel_smoke'/'2010'
args=json.loads((case/'env_args.json').read_text())
env=gym.make('gym_dssat_pdi:GymDssatPdi-v0', **args).unwrapped
print('raw action_space', env.action_space)
print('action names', list(env.action_space.keys()) if isinstance(env.action_space, dict) else env.action_space)
print('observation vars', getattr(env,'observation_variables',None))
obs=env.reset()
print('reset type', type(obs), obs if isinstance(obs, dict) else 'non-dict')
print('history keys', getattr(env,'history',{}).keys() if hasattr(env,'history') else None)
print('tmp', getattr(env,'_tmp_folder',None))
env.close()
