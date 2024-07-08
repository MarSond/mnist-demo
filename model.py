import torch
import torch.nn as nn


class MnistModel(nn.Module):
	def __init__(self):
		super(MnistModel, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.fc1 = nn.Linear(64*5*5, 128)
		self.fc2 = nn.Linear(128, 10)
		self.sequential = nn.Sequential(
			self.conv1,
			nn.ReLU(),
			nn.MaxPool2d(2),
			self.conv2,
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Flatten(),
			self.fc1,
			nn.ReLU(),
			self.fc2
		)
		
	def forward(self, x):
		x = self.sequential(x)
		return x