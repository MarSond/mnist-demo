import torch
import torch.nn as nn


class MnistModel(nn.Module):
	def __init__(self):
		super(MnistModel, self).__init__()
		self.linear = nn.Linear(784, 10)
		
	def forward(self, x):
		x = x.view(-1, 784)
		return self.linear(x)