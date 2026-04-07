import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class RNDVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, rnd_model, beta=0.05):
        super().__init__(venv)
        self.rnd = rnd_model
        self.beta = beta

    def reset(self):
        obs = self.venv.reset()
        self.rnd.obs_rms.update(obs)
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # update observation normalization
        self.rnd.obs_rms.update(obs)

        norm_obs = self.rnd.normalize_obs(obs)

        intrinsic = self.rnd.compute_intrinsic(norm_obs)

        # normalize intrinsic reward
        self.rnd.int_rms.update(intrinsic)
        intrinsic_norm = intrinsic / np.sqrt(self.rnd.int_rms.var + 1e-8)

        total_rewards = rewards + self.beta * intrinsic_norm

        # logging
        for i, info in enumerate(infos):
            info["intrinsic_reward"] = intrinsic[i]
            info["extrinsic_reward"] = rewards[i]
            info["total_reward"] = total_rewards[i]

        return obs, total_rewards, dones, infos
