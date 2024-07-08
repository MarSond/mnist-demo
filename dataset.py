import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class MnistDataset(Dataset):
	def __init__(self, datalist):
		self.data = datalist["file_path"].values
		self.target = datalist["label"].values
		
	def __getitem__(self, index):
		path = self.data[index]
		# load image
		image = Image.open(path)
		# convert to tensor
		input = torch.tensor(np.array(image)).float()
		label = self.target[index]
		return input, label
	
	def __len__(self):
		return len(self.data)