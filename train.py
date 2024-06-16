import pandas as pd
import numpy as np
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import MnistDataset
from model import MnistModel


"""
Wir brauchen:
- Datenliste mit pfad und label und metadaten
- Eine Modellstruktur
-- Layer, ausgabe, input
- Dataset Klasse
-- Genutzt vom DataLoader
- Training/validation loop
-- Training in Training/Validation split
-- Testdatensatz unabhängig am Ende
-- Epochs
Metriken
Model speichern

Inferenz mit testdatensatz
- Model laden
Eigene Bilder testen

Weiteres zu erklären:
Seeds setzen
Optimierer
Loss
Batchsize
Augmentierung
"""

class Training:


	def setup():
		np.random.seed(42)
		torch.manual_seed(42)
		assert torch.cuda.is_available(), "CUDA is not available"
		torch.cuda.manual_seed_all(42)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


	def get_dataset() -> pd.DataFrame:
		"""
		Load the dataset from the data folder.
		Example filename  : data/mnist_train/0_512.png -> label 0
		"""
		file_list = glob.glob("data/mnist_train/*.png")
		data = pd.DataFrame()
		data["file_path"] = file_list
		data["label"] = data["file_path"].apply(lambda x: int(os.path.basename(x).split("_")[0]))


		data = data.sample(frac=1).reset_index(drop=True) # Shuffle the dataset
		print(f"Dataset loaded with {len(data)} samples")
		print(data.head())
		# 80% train, 20% validation
		split = int(0.8 * len(data))
		train_dataset = data.iloc[:split]
		valid_dataset = data.iloc[split:]

		train_dataset = MnistDataset(train_dataset)
		valid_dataset = MnistDataset(valid_dataset)

		return train_dataset, valid_dataset

	def prepare(self):
		train_dataset, valid_dataset = self.get_dataset()

		self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
		self.valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

	def start(self):
		self.setup()

		self.prepare()

		self.model = MnistModel()


		
		# Training loop
		for i, (input, label) in enumerate(train_loader):
			print(input.shape, label.shape)
			break 

		# validation loop
		for i, (input, label) in enumerate(valid_loader):
			print(input.shape, label.shape)
			break




if __name__ == "__main__":
	train = Training()
	train.start()
	