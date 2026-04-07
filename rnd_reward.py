import torch
from torch import nn

def init_orthogonal(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class RNDIntrinsicReward:
    def __init__(self, obs_shape):
        self.target = RandomNetwork(obs_shape)
        self.predictor = PredictorNetwork(obs_shape)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)

    def compute(self, obs):
        """Return intrinsic reward (float or np.array)."""
        with torch.no_grad():
            target = self.target(obs)
        pred = self.predictor(obs)
        return ((pred - target) ** 2).mean(dim=-1)

    def update(self, obs):
        target = self.target(obs).detach()
        pred = self.predictor(obs)
        loss = ((pred - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ----------------------------
# Target Network (Random, fixed)
# ----------------------------
class RandomNetwork(nn.Module):
    def __init__(self, input_shape, output_size=512):
        super().__init__()
        c, h, w = input_shape  # channels, height, width

        # CNN layers similar to DQN paper
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # compute conv output size
        def conv2d_size_out(size, kernel_size=1, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Linear(linear_input_size, output_size)

        # freeze parameters (target network is fixed)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ----------------------------
# Predictor Network (trainable)
# ----------------------------
class PredictorNetwork(nn.Module):
    def __init__(self, input_shape, output_size=512):
        super().__init__()
        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        def conv2d_size_out(size, kernel_size=1, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Linear(linear_input_size, output_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
