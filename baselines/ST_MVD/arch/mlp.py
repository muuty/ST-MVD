import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Residual MLP block implemented with 1x1 Conv2d.

    Operates on (B, D, N, 1) tensors so that node and time dims are preserved.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.fc2(self.drop(self.act(self.fc1(x))))
