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
			{'params': [p for n, p in classification_model.named_parameters() if not any(nd in n for nd in no_decay)],
			'weight_decay': 0.01},
			{'params': [p for n, p in classification_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

		for epoch_index in range(config.n_epoch):
			print("[epoch {}]\n".format(epoch_index + 1))

			losses = []
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
				losses.append(loss.item())
				loss.backward()
				optimizer.step()
			end_time = time.time()
			train_loss = np.mean(losses)
			
			trai_temp_loss, train_acc = self.test_model(classification_model, train_dataset, train_loader)
			print("train: acc {} / loss {} / time {}".format(train_acc, train_loss, end_time - start_time))

			test_loss, test_acc = self.test_model(classification_model, test_dataset, test_loader)
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
	
	def test_model(self, model, test_dataset, test_loader):

		loss = 0
		acc = 0

		model.eval()
		for data in test_loader:
			with torch.no_grad():
				inputs = {
					'input_ids': data['input_ids'],
					'attention_mask': data['attention_mask'],
					'labels': data['labels']
				}
				outputs = model(**inputs)
				loss += outputs[0]
				logit = outputs[1]
				acc += (logit.argmax(1)==inputs['labels']).sum().item()
		
		return loss / len(test_dataset), acc / len(test_dataset)

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

    sliced_datas = []

    for zd in zipped_data:
      d = []

      if len(zd[0]) > max_seq_len:
        d.append(zd[0][:max_seq_len])
      else:
        d.append(zd[0])
      
      d.append(zd[1])
      sliced_datas.append(d)

    for sliced_data in sliced_datas:
      index_of_words = self.tokenizer.encode(sliced_data[0])
      token_type_ids = [0] * len(index_of_words)
      attention_mask = [1] * len(index_of_words)

      # Padding Length
      padding_length = max_seq_len - len(index_of_words)

      # Zero Padding
      index_of_words += [0] * padding_length
      token_type_ids += [0] * padding_length
      attention_mask += [0] * padding_length

      # Label
      label = int(sliced_data[1])
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