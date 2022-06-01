from timeit import default_timer as timer
import gym
import numpy as np
import warnings
import pdb

if __name__ == '__main__':
    humanoidstandup = gym.make('HumanoidStandup-v2')
    humanoidstandup.reset()
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': './logs/dssat_pdi.log',
        'mode': 'fertilization',
        'seed': 123456,
        'random_weather': True,
    }
    gymDSSAT = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    gymDSSAT.reset()
    repetitions = 1000
    steps = 100

    print(f'Measuring step time for {steps} time steps, {repetitions} repetitions for HumanoidStandup-v2')
    humanoidstandup_times = []
    for repetition in range(repetitions):
        if (repetition + 1) % 100 == 0:
            print(f'repetition {repetition+1}/{repetitions}')
        for _ in range(steps):
            action = humanoidstandup.action_space.sample()
            start = timer()
            _, _, done, _ = humanoidstandup.step(action)
            end = timer()
            if done:
                warnings.warn('Warning: the episode has early ended for HumanoidStandup-v2')
            humanoidstandup_times.append(end-start)  # the time difference is in seconds by default
        humanoidstandup.reset()
    humanoidstandup.close()

    print(f'HumanoidStandup-v2 mean step time {1000 * np.mean(humanoidstandup_times)} milliseconds, std {1000 * np.std(humanoidstandup_times, ddof=1)} milliseconds')

    print(f'Measuring step time for {steps} time steps, {repetitions} repetitions for GymDssatPdi-v0')
    gymDSSAT_times = []
    for repetition in range(repetitions):
        if (repetition + 1) % 100 == 0:
            print(f'repetition {repetition+1}/{repetitions}')
        for _ in range(steps):
            action = dict(gymDSSAT.action_space.sample())
            action = {key: action[key].item() for key in [*action]}
            start = timer()
            _, _, done, _ = gymDSSAT.step(action)
            end = timer()
            if done:
                warnings.warn('Warning: the episode has early ended for gymDSSAT')
            gymDSSAT_times.append(end-start)
        gymDSSAT.reset()
    gymDSSAT.close()

    print(f'gymDSSAT mean step time {1000 * np.mean(gymDSSAT_times)} milliseconds, std {1000 * np.std(gymDSSAT_times, ddof=1)} milliseconds')