import pandas as pd
import numpy as np
import torch
import random
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from .model.koelectra_classifier import KoElectraClassifier
from tqdm.notebook import tqdm
import torch
import os
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tqdm, tqdm_notebook
from kobert_transformers import get_tokenizer
from transformers import (
	ElectraConfig,
)
import time

class KoElectraClassificationTrainer:
	def __init__(self):
		pass

	def train(self, train_data, train_label, test_data, test_label, config, device, model_output_path):
		electra_config = ElectraConfig.from_pretrained("monologg/koelectra-small-v2-discriminator")
		classification_model = KoElectraClassifier.from_pretrained(pretrained_model_name_or_path = "monologg/koelectra-small-v2-discriminator", config = electra_config, num_labels = config.num_label)
		tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

		train_zipped_data = make_zipped_data(train_data, train_label)
		test_zipped_data = make_zipped_data(test_data, test_label)
		learning_rate = config.learning_rate

		classification_model.to(device)

		train_dataset = KoElectraClassificationDataset(tokenizer=tokenizer, device=device, zipped_data=train_zipped_data, max_seq_len = config.max_seq_len)
		test_dataset = KoElectraClassificationDataset(tokenizer=tokenizer, device=device, zipped_data=test_zipped_data, max_seq_len = config.max_seq_len)

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{
				'params': [p for n, p in classification_model.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': 0.01
			},
			{
				'params': [p for n, p in classification_model.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0
			},
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

		for epoch_index in range(config.n_epoch):
			print("[epoch {}]\n".format(epoch_index + 1))

			train_losses = []
			train_acc = 0
			classification_model.train()
			start_time = time.time()
			for batch_index, data in enumerate(tqdm_notebook(train_loader)):
				optimizer.zero_grad()
				inputs = {
					'input_ids': data['input_ids'],
					'attention_mask': data['attention_mask'],
					'labels': data['labels']
				}
				outputs = classification_model(**inputs)
				loss = outputs[0]
				logit = outputs[1]
				train_losses.append(loss.item())
				loss.backward()
				optimizer.step()
				train_acc += (logit.argmax(1)==inputs['labels']).sum().item()
			end_time = time.time()
			train_loss = np.mean(train_losses)
			train_acc = train_acc / len(train_dataset)
			print("train: acc {} / loss {} / time {}".format(train_acc, train_loss, end_time - start_time))

			test_losses = []
			test_acc = 0
			classification_model.eval()
			for batch_index, data in enumerate(test_loader):
				with torch.no_grad():
					inputs = {
						'input_ids': data['input_ids'],
						'attention_mask': data['attention_mask'],
						'labels': data['labels']
					}
					outputs = classification_model(**inputs)
					loss = outputs[0]
					logit = outputs[1]
					test_losses.append(loss.item())
					test_acc += (logit.argmax(1)==inputs['labels']).sum().item()
					print('hihi', logit.argmax(1), inputs['labels'])
			
			test_loss = np.mean(test_losses)
			test_acc = test_acc / len(test_dataset)
			print("test: acc {} / loss {}".format(test_acc, test_loss))
			print("\n")

		torch.save({
			'epoch': config.n_epoch,  # 현재 학습 epoch
			'model_state_dict': classification_model.state_dict(),  # 모델 저장
			'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
			'loss': loss.item(),  # Loss 저장
			'train_step': config.n_epoch * config.batch_size,  # 현재 진행한 학습
			'total_train_step': len(train_loader)  # 현재 epoch에 학습 할 총 train step
		}, model_output_path)

def make_zipped_data(data, label):      
	zipped_data = []

	for i in range(len(data)):
		row = []
		row.append(data[i])
		row.append(label[i])
		zipped_data.append(row)

	return zipped_data

class KoElectraClassificationDataset(Dataset):
	def __init__(self,
							device = None,
							tokenizer = None,
							zipped_data = None,
							max_seq_len = None, # KoBERT max_length
							):

		self.device = device
		self.data =[]
		self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()

		for zd in zipped_data:
			index_of_words = self.tokenizer.encode(zd[0])

			if len(index_of_words) > max_seq_len:
				index_of_words = index_of_words[:max_seq_len]

			token_type_ids = [0] * len(index_of_words)
			attention_mask = [1] * len(index_of_words)

			# Padding Length
			padding_length = max_seq_len - len(index_of_words)

			# Zero Padding
			index_of_words += [0] * padding_length
			token_type_ids += [0] * padding_length
			attention_mask += [0] * padding_length

			# Label
			label = int(zd[1])
			data = {
				'input_ids': torch.tensor(index_of_words).to(self.device),
				'token_type_ids': torch.tensor(token_type_ids).to(self.device),
				'attention_mask': torch.tensor(attention_mask).to(self.device),
				'labels': torch.tensor(label).to(self.device)
			}

			self.data.append(data)

	def __len__(self):
		return len(self.data)
	def __getitem__(self,index):
		item = self.data[index]
		return item

def calc_accuracy(X,Y):
	max_vals, max_indices = torch.max(X, 1)
	train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
	return train_acc
