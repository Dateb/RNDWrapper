from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RNDTrainingCallback(BaseCallback):
    def __init__(self, rnd_model, batch_size=256, update_proportion=0.25, verbose=0):
        super().__init__(verbose)
        self.rnd = rnd_model
        self.batch_size = batch_size
        self.update_proportion = update_proportion

    def _on_rollout_end(self) -> bool:
        rollout_buffer = self.model.rollout_buffer

        # (n_steps, n_envs, obs_dim) → flatten
        obs = rollout_buffer.observations
        obs = obs.reshape(-1, *obs.shape[2:])

        # normalize (IMPORTANT: use same RMS as wrapper)
        norm_obs = self.rnd.normalize_obs(obs)

        n_samples = len(norm_obs)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]

            batch = norm_obs[batch_idx]

            # Subsampling
            mask = np.random.rand(len(batch)) < self.update_proportion
            if mask.sum() == 0:
                continue

            batch = batch[mask]

            self.rnd.update(batch)

        return True

    def _on_step(self) -> bool:
        return True