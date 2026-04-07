import numpy as np
import torch
import torch.optim as optim

from normalization import RunningMeanStd
from rnd_reward import RandomNetwork, PredictorNetwork, init_orthogonal


class RNDModel:
    def __init__(self, obs_shape, device="cpu", lr=1e-4):
        self.device = device

        torch.manual_seed(7)
        self.target = RandomNetwork(obs_shape).to(device)

        torch.manual_seed(22)
        self.predictor = PredictorNetwork(obs_shape).to(device)

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

        # normalization
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.int_rms = RunningMeanStd(shape=())

    def normalize_obs(self, obs):
        normalized = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized, -5, 5)

    def compute_intrinsic(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        pred = self.predictor(obs_t)
        with torch.no_grad():
            target = self.target(obs_t)

        error = ((pred - target) ** 2).mean(dim=1).detach().cpu().numpy()

        return error

    def update(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        pred = self.predictor(obs_t)
        with torch.no_grad():
            target = self.target(obs_t)

        loss = ((pred - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
