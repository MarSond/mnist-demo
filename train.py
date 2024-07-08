import pandas as pd
import numpy as np
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import MnistDataset
from model import MnistModel
from tqdm import tqdm
from PIL import Image
import metrics
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


	def setup(self):
		np.random.seed(42)
		torch.manual_seed(42)
		assert torch.cuda.is_available(), "CUDA is not available"
		torch.cuda.manual_seed_all(42)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


	def get_dataset(self) -> pd.DataFrame:
		"""
		Load the dataset from the data folder.
		Example filename  : data/mnist_train/0_512.png -> label 0
		"""
		file_list = glob.glob("data/mnist_train/*.png")
		data = pd.DataFrame()
		data["file_path"] = file_list
		data["label"] = data["file_path"].apply(lambda x: int(os.path.basename(x).split("_")[0]))


		data = data.sample(frac=0.4).reset_index(drop=True) # Shuffle the dataset
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
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Using device: {self.device}")
		self.model = MnistModel()	# Fresh model
		self.model = self.model.to(self.device)
		train_dataset, valid_dataset = self.get_dataset()

		self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
		self.valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

		self.criterion = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		self.metrics = metrics.MetricsTracker(
				metrics_class=metrics.TorchMetricsAdapter, num_classes=10, device=self.device)

	def save_model(self):
		torch.save(self.model.state_dict(), "model.pth")


	def start_epochs(self, epochs):
		print("Starting training with {} epochs".format(epochs))
		
		for epoch in range(epochs):
			# training
			self.model.train()
			self.metrics.prepare_new_epoch(validation=True)
			print(f"Starting epoch {epoch}")
			train_progress_bar = tqdm(self.train_loader, desc="Training")
			for inputs, labels in train_progress_bar:
				inputs = inputs.unsqueeze(1)
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)
				# print inputs as image
				img1 = inputs[0].cpu().numpy()
				img2 = inputs[1].cpu().numpy()
				#Image.fromarray(img1).show()
				#Image.fromarray(img2).show()
				#break
				
				#print(f"Inputs: {inputs.shape}")
				self.optimizer.zero_grad()
				outputs = self.model(inputs)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()
				self.metrics.update_step(probabilities=outputs, labels=labels, loss=loss, validation=False)
				
			train_epoch_metrics = self.metrics.save_epoch_metrics(validation=False)
			train_epoch_metrics[metrics.Names.ROC_DATA] = []
			print(f"Training metrics: {train_epoch_metrics}")


			# validation
			self.metrics.prepare_new_epoch(validation=True)
			self.model.eval()
			prediction_list = []
			label_list = []
			valid_progress_bar = tqdm(self.valid_loader, desc="Validation")
			for inputs, labels in valid_progress_bar:
				inputs = inputs.unsqueeze(1)
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)
				outputs = self.model(inputs)
				loss = self.criterion(outputs, labels)
				self.metrics.update_step(probabilities=outputs, labels=labels, loss=loss, validation=True)
			valid_epoch_metrics = self.metrics.save_epoch_metrics(validation=True)
			valid_epoch_metrics[metrics.Names.ROC_DATA] = []
			print(f"Validation metrics: {valid_epoch_metrics}")
		self.metrics.finish_fold()

			
	def start(self):
		self.setup()

		self.prepare()

		

		# Training loop
		self.start_epochs(2)

		print("Finished training")
		self.metrics.print_end_summary()
		self.metrics.plot_metrics()
		self.save_model()
		
		


if __name__ == "__main__":
	train = Training()
	train.start()
	