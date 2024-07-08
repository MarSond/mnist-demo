import json
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
import torchmetrics as tm
import torch
import tabulate
from os.path import join
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, roc_curve, roc_auc_score
from torch import Tensor
import matplotlib.pyplot as plt
try:
	from run import Run
	standalone = False
except ImportError:
	print("Metrics standalone mode")
	standalone = True

class NumpyJsonEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

class Names:
	TRAINING	=	"training"
	VALIDATION	=	"validation"
	ACCURACY	=	"acc"
	F1			=	"f1"
	PRECISION	=	"precision"
	RECALL		=	"recall"
	CONFUSION	=	"confusion"
	SPECIFICITY	=	"specificity"
	MCC			=	"mcc"
	NMCC		=	"nmcc"
	ROC_DATA	=	"roc_data"
	AUROC 		=	"auroc"
	MEAN_LOSS	=	"mean_loss"
	EPOCH		=	"epoch"
	FOLD		=	"fold"
	MULTICLASS	=	"multiclass"
	PLR 		=	"plr"
	NLR 		=	"nlr"
	BINARY		=	"binary"
	MODE		= 	"mode"
	FPR			=	"fpr"
	TPR			=	"tpr"
	THRESHOLDS	=	"thresholds"
	TRAINING_EPOCHS		= "training_epochs"
	VALIDATION_EPOCHS	= "validation_epochs"
	FILENAME_METRICS_VALUE = "metrics.json"


class MetricsInterface(ABC):

	def __init__(self, task_type, num_classes, device=None) -> None:
		super().__init__()
		self.task_type = task_type
		self.num_classes = num_classes
		self.device = device
		self.predictions = []
		self.labels = []
		self.loss = []
		self.probabilities = []
		self.one_update_done = False

	def calculate_and_store_roc(self):
		desired_thresholds = np.linspace(0, 1, 100)  # feste Schwellenwerte der ROC Kurve
		auroc_data = []
		fpr_data = []
		tpr_data = []

		if self.num_classes == 2:  # Binary classification
			y_score = np.array(self.probabilities)[:, 1]
			fpr, tpr, thresholds = roc_curve(self.labels, y_score)
			
			# Interpolation
			interpolated_fpr = np.interp(desired_thresholds, thresholds[::-1], fpr[::-1])
			interpolated_tpr = np.interp(desired_thresholds, thresholds[::-1], tpr[::-1])

			fpr_data.append(interpolated_fpr.tolist())
			tpr_data.append(interpolated_tpr.tolist())
			auroc_data = roc_auc_score(self.labels, y_score)
		else:  # Multi-class classification - untested
			for i in range(self.num_classes):
				y_true_i = (np.array(self.labels) == i).astype(int)
				y_score_i = np.array(self.probabilities)[:, i]
				fpr, tpr, thresholds = roc_curve(y_true_i, y_score_i)
				
				# Interpolation
				interpolated_fpr = np.interp(desired_thresholds, thresholds[::-1], fpr[::-1])
				interpolated_tpr = np.interp(desired_thresholds, thresholds[::-1], tpr[::-1])
				
				fpr_data.append(interpolated_fpr.tolist())
				tpr_data.append(interpolated_tpr.tolist())
				auroc_data.append(roc_auc_score(y_true_i, y_score_i))

		roc_data = {
			Names.FPR: fpr_data,
			Names.TPR: tpr_data,
			Names.THRESHOLDS: desired_thresholds.tolist(),
			Names.AUROC: auroc_data,
		}
		return roc_data
	
	def _get_fake_update_data(self, num=2):
		if num == 3:
			probs_0 = np.array([[0.6, 0.2, 0.2]] * 20)
			probs_1 = np.array([[0.1, 0.7, 0.2]] * 30)
			probs_2 = np.array([[0.1, 0.1, 0.8]] * 50)
			fake_probabilities = np.vstack([probs_0, probs_1, probs_2])
			# Für die Vorhersagen nehmen wir an, dass:
			# - 10 der Klasse-0-Instanzen richtig klassifiziert wurden (TP für Klasse 0)
			# - 20 der Klasse-1-Instanzen richtig klassifiziert wurden (TP für Klasse 1)
			# - 40 der Klasse-2-Instanzen richtig klassifiziert wurden (TP für Klasse 2)

			# Die restlichen Vorhersagen sind zufällig falsch, um die Matrix interessanter zu gestalten.
			# Angenommen, die falschen Vorhersagen sind wie folgt verteilt:
			# - 10 der Klasse-0-Instanzen wurden als Klasse 1 klassifiziert (FP für Klasse 1)
			# - 10 der Klasse-1-Instanzen wurden als Klasse 2 klassifiziert (FP für Klasse 2)
			# - 10 der Klasse-2-Instanzen wurden als Klasse 0 klassifiziert (FP für Klasse 0)
			fake_labels = [0]*20 + [1]*30 + [2]*50
			fake_predictions = [0]*10 + [1]*10 + [1]*20 + [2]*10 + [2]*40 + [0]*10

		elif num == 2:
			fake_labels = [1]*10 + [0]*40 + [1]*20 + [0]*30
			fake_predictions = [1]*10 + [1]*40 + [0]*20 + [0]*30
			# sum = 100 # total 1: 30, total 0: 70
			# tp = 10
			# fp = 40
			# fn = 20
			# tn = 30
			# Ziel: [TP, FP; FN, TN]
			# [10, 40; 20, 30]
			fake_probabilities = np.array([[0.6, 0.4]] * 10 + [[0.3, 0.7]] * 40 + [[0.6, 0.4]] * 20 + [[0.3, 0.7]] * 30)
		return fake_labels, fake_predictions, fake_probabilities


	# update the history each batch with new predictions and labels
	@abstractmethod
	def update(self, probabilities, labels, loss=None):
		pass

	# compute the scores based on the history
	@abstractmethod
	def compute(self):
		pass

	# delete all history, ready for a new fold/epoch
	def reset(self):
		self.predictions = []
		self.labels = []
		self.loss = []
		self.probabilities = []

class SKLearnMetricsAdapter(MetricsInterface):
	def __init__(self, num_classes, task_type, device):
		super().__init__(task_type, num_classes, device)
		self.num_classes = 2
		self.task_type = "binary"
		metrics_collection = tm.MetricCollection({
			Names.ACCURACY: tm.Accuracy(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.F1: tm.F1Score(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.PRECISION: tm.Precision(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.RECALL: tm.Recall(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.CONFUSION: tm.ConfusionMatrix(num_classes=self.num_classes, normalize="none", threshold=0.5, task=self.task_type),
			Names.SPECIFICITY: tm.Specificity(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.MCC: tm.MatthewsCorrCoef(num_classes=self.num_classes, task=self.task_type),
		})
		self.loss = []
		self.labels = []
		self.probabiities = []
		self.metrics_collection = metrics_collection
		# TODO compare update since it is same when using both at the same time
	def update(self, probabilities, labels, loss=None, num_fake=0):
		if self.one_update_done:
			return
		# move to cpu if necessary
		if probabilities.device != torch.device('cpu'):
			probabilities = probabilities.cpu()
		if labels.device != torch.device('cpu'):
			labels = labels.cpu()
		if loss is not None and loss.device != torch.device('cpu'):
			loss = loss.cpu()

		predictions = probabilities.argmax(axis=1)

		if num_fake > 0:
			fake_labels, fake_predictions, fake_probabilities = self._get_fake_update_data(num_fake)
			predictions = fake_predictions
			labels = fake_labels
			probabilities = fake_probabilities
			self.one_update_done = True
		self.metrics_collection.update(predictions, labels)
		self.predictions.extend(predictions)
		self.labels.extend(labels)
		self.probabilities.extend(probabilities)
		if loss is not None:
			self.loss.append(loss)
	 # TODO Why is F1 score and precision wrong? TP / (TP/FP) = 0.6 but here is is 0.779
	from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef

	def compute(self):
		metrics = {}

		# Using 'binary' for binary classification if needed
		average_method = 'macro' if self.num_classes > 2 else 'binary'

		# Convert lists to numpy arrays
		labels_array = np.array(self.labels)
		predictions_array = np.array(self.predictions)

		# Sklearn Metrics
		if self.num_classes == 2:
			pass
		metrics[Names.ACCURACY] = accuracy_score(labels_array, predictions_array)
		metrics[Names.F1] = f1_score(labels_array, predictions_array, average=average_method)
		metrics[Names.PRECISION] = precision_score(labels_array, predictions_array, average=average_method)
		metrics[Names.RECALL] = recall_score(labels_array, predictions_array, average=average_method)
		metrics[Names.CONFUSION] = confusion_matrix(labels_array, predictions_array, normalize=None)

		tn, fp, fn, tp = metrics[Names.CONFUSION].ravel()
		metrics[Names.SPECIFICITY] = tn / (tn + fp)
		metrics[Names.MCC] = matthews_corrcoef(labels_array, predictions_array)
		metrics[Names.NMCC] = (metrics[Names.MCC] + 1.0) / 2.0
		metrics[Names.ROC_DATA] = self.calculate_and_store_roc()
		metrics[Names.AUROC] = metrics[Names.ROC_DATA][Names.AUROC]
		metrics[Names.PLR] = metrics[Names.RECALL] / (1 - metrics[Names.SPECIFICITY] + 1e-8)
		metrics[Names.NLR] = metrics[Names.PRECISION] / (1 - metrics[Names.SPECIFICITY] + 1e-8)
		# Torchmetrics
		torchmetrics_result = self.metrics_collection.compute()

		# Compare metrics
		print("Sklearn Accuracy:", metrics[Names.ACCURACY])
		print("Torchmetrics Accuracy:", torchmetrics_result[Names.ACCURACY].item())

		print("Sklearn F1:", metrics[Names.F1])
		print("Torchmetrics F1:", torchmetrics_result[Names.F1].item())

		print("Sklearn Precision:", metrics[Names.PRECISION])
		print("Torchmetrics Precision:", torchmetrics_result[Names.PRECISION].item())

		print("Sklearn Recall:", metrics[Names.RECALL])
		print("Torchmetrics Recall:", torchmetrics_result[Names.RECALL].item())

		print("Sklearn Confusion Matrix:", metrics[Names.CONFUSION])
		print("Torchmetrics Confusion Matrix:", torchmetrics_result[Names.CONFUSION].cpu().numpy())

		print("Sklearn Specificity:", metrics[Names.SPECIFICITY])
		print("Torchmetrics Specificity:", torchmetrics_result[Names.SPECIFICITY].item())

		print("Sklearn MCC:", metrics[Names.MCC])
		print("Torchmetrics MCC:", torchmetrics_result[Names.MCC].item())

		if len(self.loss) > 0:
			metrics[Names.MEAN_LOSS] = np.mean(self.loss)
		else:
			metrics[Names.MEAN_LOSS] = 0.0

		return metrics


	def reset(self):
		super().reset()

class TorchMetricsAdapter(MetricsInterface):
	def __init__(self, num_classes, task_type, device):
		super().__init__(task_type, num_classes, device)
		assert self.device is not None, "Device must be set for TorchMetricsAdapter"
		self.num_classes = num_classes
		self.task_type = task_type
		metrics_collection = tm.MetricCollection({
			Names.ACCURACY: tm.Accuracy(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.F1: tm.F1Score(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.PRECISION: tm.Precision(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.RECALL: tm.Recall(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.CONFUSION: tm.ConfusionMatrix(num_classes=self.num_classes, normalize="none", threshold=0.5, task=self.task_type),
			Names.SPECIFICITY: tm.Specificity(num_classes=self.num_classes, average='macro', task=self.task_type),
			Names.MCC: tm.MatthewsCorrCoef(num_classes=self.num_classes, task=self.task_type),
		})
		self.loss = []
		self.labels = []
		self.probabiities = []
		self.metrics_collection = metrics_collection.to(self.device)

	def _fake_step(self, num=2):
		fake_labels, fake_predictions, fake_probabilities = self._get_fake_update_data(num)

		# Umwandlung der Listen in PyTorch-Tensoren
		fake_labels = Tensor(fake_labels).to(self.device)
		fake_predictions = Tensor(fake_predictions).to(self.device)
		self.metrics_collection.update(fake_predictions, fake_labels)
		self.labels.extend(fake_labels.cpu().numpy())
		self.probabiities.extend(fake_probabilities)

	def update(self, probabilities, labels, loss=None):
		predictions = probabilities.argmax(dim=1)
		probabilities = probabilities.detach()
		
		fake = False # set to True to test the metrics with fake data
		if fake:
			self._fake_step(3)
		else:	
			self.metrics_collection.update(predictions, labels)
			self.labels.extend(labels.cpu().numpy())
			self.probabilities.extend(probabilities.cpu().numpy())
		if loss is not None:
			if isinstance(loss, Tensor):
				loss = loss.item()
			self.loss.append(loss)

	def compute(self):
		metrics: dict = self.metrics_collection.compute()
		# Umstellen der Elemente, um das gewünschte Format zu erhalten [TP, FP; FN, TN]
		# https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html#binaryconfusionmatrix
		tm_conv = metrics[Names.CONFUSION].cpu().numpy() # CM von Torchmetrics
		if self.task_type == Names.BINARY:
			target_matrix = tm_conv.T[:, ::-1][::-1, :] # Torchmetrics 2 class
		else: 
			target_matrix = tm_conv
		# normalize MCC to [0-1]
		metrics[Names.NMCC] = (metrics[Names.MCC] / + 1.0) / 2.0
		metrics[Names.CONFUSION] = target_matrix
		metrics[Names.ROC_DATA] = self.calculate_and_store_roc()
		metrics[Names.AUROC] = metrics[Names.ROC_DATA][Names.AUROC]
		metrics[Names.PLR] = metrics[Names.RECALL] / (1 - metrics[Names.SPECIFICITY] + 1e-8) 
		metrics[Names.NLR] = (1 - metrics[Names.RECALL]) / (metrics[Names.SPECIFICITY] + 1e-8)

		if len(self.loss) > 0:
			metrics[Names.MEAN_LOSS] = np.mean(self.loss)
		else: 
			metrics[Names.MEAN_LOSS] = 0.0
		return metrics
	
	def reset(self):
		super().reset()
		self.metrics_collection.reset()


class MetricsTracker:
	# run optional
	def __init__(self, metrics_class, run=None, device = None, logger=None, fl_precision=4, num_classes=None):
		if logger is None:
			print("Using default logger")
			self.logger = logging.getLogger()
		else:
			self.logger = logger
			
		if run is not None:
			self.job_base_path = run.run_results_path
			self.num_classes = run.task.dataset.num_classes
		else:
			self.job_base_path = "./"  # Set a default 
			if num_classes is None:
				raise ValueError("num_classes must be set if run is None")
			self.num_classes = num_classes  # Set a default or handle it accordingly

		self.epoch_train_history = []
		self.epoch_valid_history = []
		self.fold_history = []
		self.fl_precision = fl_precision
		self.all_data_df = pd.DataFrame()
		if self.num_classes > 2:
			self.task_type = Names.MULTICLASS
		else:
			self.task_type = Names.BINARY

		self.train_metrics = metrics_class(self.num_classes, self.task_type, device)
		self.valid_metrics = metrics_class(self.num_classes, self.task_type, device)

	def update_step(self, validation, probabilities, labels, loss=None):
		if validation:
			self.valid_metrics.update(probabilities, labels, loss)
		else:
			self.train_metrics.update(probabilities, labels, loss)

	def prepare_new_epoch(self, validation):
		# resets the metrics for the new epoch and checks that no unsaved history exists
		if validation:
			self.valid_metrics.reset()
		else:
			self.train_metrics.reset()


	# TODO Epoch 0 vs 1 start
	def save_epoch_metrics(self, validation=False) -> dict:
		if validation:
			epoch_metrics = self.valid_metrics.compute()
			epoch_metrics_dict = self._get_dict(epoch_metrics)
			epoch_metrics_dict[Names.EPOCH] = len(self.epoch_valid_history) + 1
			self.epoch_valid_history.append(epoch_metrics_dict)
		else:
			epoch_metrics = self.train_metrics.compute()
			epoch_metrics_dict = self._get_dict(epoch_metrics)
			epoch_metrics_dict[Names.EPOCH] = len(self.epoch_train_history) + 1 
			self.epoch_train_history.append(epoch_metrics_dict)
		return epoch_metrics_dict

	def _reset_epoch_history(self):
		self.epoch_train_history = []
		self.epoch_valid_history = []
		self.train_metrics.reset()
		self.valid_metrics.reset()

	def finish_fold(self):
		train_valid_metrics = {
			Names.TRAINING_EPOCHS: self.epoch_train_history,
			Names.VALIDATION_EPOCHS: self.epoch_valid_history,
			Names.FOLD: len(self.fold_history) + 1
		}
		self.fold_history.append(train_valid_metrics)
		for epoch_metrics in self.epoch_train_history:
			epoch_metrics[Names.FOLD] = len(self.fold_history)
			epoch_metrics[Names.MODE] = Names.TRAINING
			self.all_data_df = pd.concat([self.all_data_df, pd.DataFrame([epoch_metrics])], axis=0)

		for epoch_metrics in self.epoch_valid_history:
			epoch_metrics[Names.FOLD] = len(self.fold_history)
			epoch_metrics[Names.MODE] = Names.VALIDATION
			self.all_data_df = pd.concat([self.all_data_df, pd.DataFrame([epoch_metrics])], axis=0)
		self.save_metrics_to_json()
		self._reset_epoch_history()


#####################


	def _truncate_floats(self, x, precision=4):
		if isinstance(x, float):
			return round(x, precision)
		elif isinstance(x, list):
			return [self._truncate_floats(xi, precision) for xi in x]
		else:
			return x

	def _get_value(self, metrics, metric_name):
		if isinstance(metrics[metric_name], Tensor):
			return float(f"{metrics[metric_name].item():.{self.fl_precision}f}")
		elif isinstance(metrics[metric_name], float):
			return float(f"{metrics[metric_name]:.{self.fl_precision}f}")
		elif isinstance(metrics[metric_name], list):
			return self._truncate_floats(metrics[metric_name], self.fl_precision)
		else:
			self.logger.error(f"Error getting value for {metric_name}. Type: {type(metrics[metric_name])}")

	def _get_dict(self, metrics):
		confusion_metric = metrics[Names.CONFUSION]
		roc_data = metrics[Names.ROC_DATA]
		roc_data = {k: self._truncate_floats(v, self.fl_precision+1) for k, v in roc_data.items()}
		if isinstance(confusion_metric, Tensor):
			confusion_metric = confusion_metric.cpu().numpy().astype(np.float32)

		confusion_round = np.round(confusion_metric, self.fl_precision)
		confusion_list	= self._truncate_floats(confusion_round.tolist(), self.fl_precision)
		metric_dict = {
			Names.ACCURACY		: self._get_value(metrics, Names.ACCURACY),
			Names.F1 			: self._get_value(metrics, Names.F1),
			Names.PRECISION 	: self._get_value(metrics, Names.PRECISION),
			Names.RECALL 		: self._get_value(metrics, Names.RECALL),
			Names.SPECIFICITY 	: self._get_value(metrics, Names.SPECIFICITY),
			Names.NMCC 			: self._get_value(metrics, Names.NMCC),
			Names.AUROC 		: self._get_value(metrics, Names.AUROC),
			Names.PLR 			: self._get_value(metrics, Names.PLR),
			Names.NLR 			: self._get_value(metrics, Names.NLR),
			Names.CONFUSION 	: confusion_list,
			Names.ROC_DATA 		: roc_data,
		}
		if len(metrics[Names.CONFUSION]) > 2:
			# Adjust format of confusion matrix for n classes
			# Not tested!
			n = len(metrics[Names.CONFUSION])
			metric_dict[Names.CONFUSION] = [metrics[Names.CONFUSION][i].tolist() for i in range(n)]

		if isinstance(metrics[Names.MEAN_LOSS], Tensor):
			metric_dict[Names.MEAN_LOSS] = self.train_metrics.loss.compute().item()
		elif isinstance(metrics[Names.MEAN_LOSS], float):
			metric_dict[Names.MEAN_LOSS] = metrics[Names.MEAN_LOSS]
		elif isinstance(metrics[Names.MEAN_LOSS], tm.MeanMetric):
			metric_dict[Names.MEAN_LOSS] = metrics[Names.MEAN_LOSS].compute().item()
		elif isinstance(metrics[Names.MEAN_LOSS], np.number):  # This will catch numpy scalars including float32
			metric_dict[Names.MEAN_LOSS] = float(metrics[Names.MEAN_LOSS])
		else:
			raise ValueError(f"Error getting mean loss: {metrics[Names.MEAN_LOSS]}")
		metric_dict[Names.EPOCH] = len(self.epoch_train_history) + 1
		metric_dict[Names.FOLD] = len(self.fold_history) + 1
		return metric_dict

	# postfix to be used in tqmd postfix after an epoch
	def get_last_validation_postfix(self):
		if len(self.epoch_valid_history) > 0:
			last_valid_epoch = self.epoch_valid_history[-1]
			# drop the epoch and fold keys + ROC data
			last_valid_epoch = {k:v for k,v in last_valid_epoch.items() if k not in [Names.EPOCH, Names.FOLD, Names.CONFUSION, Names.ROC_DATA]}
			return last_valid_epoch
		else:
			return {}

	def _prepare_result_table(self, dataframe, mode):
		# select which metrics to show in table
		metrics = [Names.ACCURACY, Names.F1, Names.PRECISION, Names.RECALL, Names.SPECIFICITY, Names.AUROC, Names.NMCC, Names.MEAN_LOSS, Names.PLR, Names.NLR]
		# TODO AUROC
		aggfunc = {metric: 'mean' for metric in metrics}
		dataframe = dataframe[dataframe[Names.MODE] == mode]
		dataframe = dataframe.drop(columns=[Names.MODE])
		# drop all not in metrics
		dataframe = dataframe.drop(columns=[col for col in dataframe.columns if col not in metrics and col not in [Names.EPOCH, Names.FOLD]])
		dataframe[metrics] = dataframe[metrics].apply(pd.to_numeric)
		dataframe = dataframe.pivot_table(index=Names.EPOCH, values=metrics, aggfunc=aggfunc)
		return dataframe

	def get_fold_averages_table(self, show_training=True):
		df = self.all_data_df
		if df.empty:
			return pd.DataFrame()

		try:
			if show_training:
				train_df = self._prepare_result_table(df, Names.TRAINING)
			else:
				train_df = pd.DataFrame()
			valid_df = self._prepare_result_table(df, Names.VALIDATION)

			# Hier verwenden wir pd.concat anstatt join
			dfs = []
			if not train_df.empty:
				train_df.columns = pd.MultiIndex.from_product([[Names.TRAINING], train_df.columns])
				dfs.append(train_df)
			if not valid_df.empty:
				valid_df.columns = pd.MultiIndex.from_product([[Names.VALIDATION], valid_df.columns])
				dfs.append(valid_df)

			result_df = pd.concat(dfs, axis=1, sort=True)
			result_df.columns = result_df.columns.map('_'.join)
			# Sortieren der Spalten
			sorted_columns = sorted(result_df.columns, key=lambda x: (x.split('_')[-1], x.split('_')[0]))
			result_df = result_df[sorted_columns]

		except ValueError as e:
			self.logger.error(f"Error computing epoch metrics: {e}")
			result_df = pd.DataFrame()
		return result_df

	def save_metrics_to_json(self):
		path = join(self.job_base_path, Names.FILENAME_METRICS_VALUE)
		try:
			with open(path, 'w') as f:
				json.dump(self.fold_history, f, cls=NumpyJsonEncoder)
			self.logger.info(f"Metrics saved to {path}")
		except Exception as e:
			self.logger.error(f"Error saving metrics: {e} to {path}")

	# plot of each run, train and valid in one plot. If kfold, then plot the averages. Train=blue, validation=orange. Do one large plot with subplots. Do it for accuarcy and f1
	def plot_metrics(self):
		
		import seaborn as sns
		sns.set_theme()
		df = self.all_data_df
		if df.empty:
			self.logger.warning("The data frame is empty. No metrics to plot.")
			return

		metrics = [
			Names.ACCURACY, Names.F1, Names.PRECISION, Names.RECALL, 
			Names.SPECIFICITY, Names.NMCC, Names.MEAN_LOSS, Names.AUROC, 
			Names.PLR, Names.NLR
		]
		aggfunc = {metric: 'mean' for metric in metrics}

		try:
			# TODO check if train is even filled - modular, allow missing values
			train_df = df[df[Names.MODE] == Names.TRAINING]
			valid_df = df[df[Names.MODE] == Names.VALIDATION]

			train_df = train_df.drop(columns=[Names.MODE])
			valid_df = valid_df.drop(columns=[Names.MODE])

			train_present = not train_df.empty
			valid_present = not valid_df.empty
			
			result_df = pd.DataFrame()
			
			if train_present:
				train_df[metrics] = train_df[metrics].apply(pd.to_numeric, errors='coerce')
				train_result_df = train_df.pivot_table(index=Names.EPOCH, values=metrics, aggfunc=aggfunc)
				train_result_df.columns = [f'{col}_train' for col in train_result_df.columns]
				result_df = train_result_df
			
			if valid_present:
				valid_df[metrics] = valid_df[metrics].apply(pd.to_numeric, errors='coerce')
				valid_result_df = valid_df.pivot_table(index=Names.EPOCH, values=metrics, aggfunc=aggfunc)
				valid_result_df.columns = [f'{col}_valid' for col in valid_result_df.columns]
				result_df = result_df.join(valid_result_df, how='outer')
				
			result_df = result_df.sort_index(axis=1)
		except ValueError as e:
			self.logger.error(f"Error computing epoch metrics: {e}")
			result_df = pd.DataFrame()
		
		if len(result_df) < 2:
			self.logger.warning("The data frame is empty or only single epoch. No metrics to plot.")
			return

		try:
			pRows = 2
			pCols = 5
			assert len(metrics) <= pRows * pCols, f"Too many metrics to plot. Max is {pRows * pCols}"
			fig, axes = plt.subplots(pRows, pCols, figsize=(17, 10))
			axes = axes.flatten()
			for i, metric in enumerate(metrics):
				train_metric = f"{metric}_train"
				valid_metric = f"{metric}_valid"
				
				if train_metric in result_df.columns:
					axes[i].plot(result_df[train_metric], label='Train')
				if valid_metric in result_df.columns:
					axes[i].plot(result_df[valid_metric], label='Validation')
					
				axes[i].set_xticks(range(len(result_df) + 1))
				axes[i].set_xlim(1, len(result_df))
				axes[i].set_xlabel("Epoch")
				axes[i].set_ylim(0, 1)
				if metric == Names.MCC:
					axes[i].set_ylim(-1, 1)
				if metric == Names.NLR or metric == Names.PLR:
					# use auto scaling
					axes[i].set_ylim(None, None)
				
				axes[i].plot(result_df[valid_metric], label=valid_metric)
				axes[i].set_title(metric)
				axes[i].legend()
				
				if i >= len(metrics):
					break

			plt.tight_layout()
			plt.savefig(join(self.job_base_path, "metrics.png"))
			plt.show(block=False)

		except Exception as e:
			self.logger.error(f"Error plotting metrics: {e}")

	
	def print_all_metrics(self, epoch=-1, validation=True) -> None:
		metrics = self.fold_history
		"""
		Print a table of average metrics and their standard deviation across all folds.
		
		Args:
			metrics (List[Dict]): List of metric dictionaries for each fold.
		"""
		
		# Initialisierung der Listen zur Speicherung der Metriken für jeden Fold
		acc, f1, precision, recall, specificity, nmcc, plr, nlr = [], [], [], [], [], [], [], []
		
		if validation:
			mode = "validation_epochs"
		else:
			mode = "training_epochs"
		
		# Sammle Metriken für jeden Fold
		for fold in metrics:
			fold_data = fold[mode]
			last_epoch = fold_data[epoch]
			
			acc.append(last_epoch["acc"])
			f1.append(last_epoch["f1"])
			precision.append(last_epoch["precision"])
			recall.append(last_epoch["recall"])
			specificity.append(last_epoch["specificity"])
			nmcc.append(last_epoch["nmcc"])
			plr.append(last_epoch["plr"])
			nlr.append(last_epoch["nlr"])
		
		# Erstelle einen DataFrame zur leichteren Darstellung
		df = pd.DataFrame({
			"Metric": ["Accuracy", "F1", "Precision", "Recall", "Specificity", "NMCC", "PLR", "NLR"],
			"Average": [np.mean(acc), np.mean(f1), np.mean(precision), np.mean(recall), np.mean(specificity), np.mean(nmcc),  np.mean(plr), np.mean(nlr)],
			"STD": [np.std(acc), np.std(f1), np.std(precision), np.std(recall), np.std(specificity), np.std(nmcc), np.std(plr), np.std(nlr)]
		})
		
		# Drucke den DataFrame in Tabellenform
		self.logger.error(f"Average metrics for all folds of epoch {epoch}:")
		self.logger.error(df.to_string(index=False))

	def print_end_summary(self):
		try:
			# set pd max float precision of self.fl_precision
			with pd.option_context('display.float_format', f'{{:.{self.fl_precision}f}}'.format):
				# pd.options.display.float_format = f'{{:.{self.fl_precision}f}}'.format
				self.logger.info("#"*50)
				#self.logger.debug(json.dumps(self.fold_history))
				
				table = self.get_fold_averages_table(show_training=False)
				self.logger.error("Averages over all folds, for every epoch:")
				self.logger.error(tabulate.tabulate(table, headers='keys', tablefmt='grid'))
				self.logger.warning("#"*50)
				self.print_all_metrics()
		except Exception as e:
			self.logger.error(f"Error printing metrics: {e}")
		#######################

	def load_state_from_json(self, path):
		with open(path, 'r') as f:
			self.fold_history = json.load(f)