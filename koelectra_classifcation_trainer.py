from transformers import AutoTokenizer, AdamW
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from IPython.display import display
from kobert_transformers import get_tokenizer
from torch.utils.data import Dataset
from .model.koelectra_classifier import KoElectraClassifier
from transformers import (
	ElectraConfig
)

class KoElectraClassficationTrainer :
	def __init__(self, config):
		electra_config = ElectraConfig.from_pretrained("monologg/koelectra-small-v2-discriminator")
		model = KoElectraClassifier.from_pretrained(pretrained_model_name_or_path = "monologg/koelectra-small-v2-discriminator", config = electra_config, num_labels = config.num_of_classes)
		tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

		self.model = model
		self.tokenizer = tokenizer
		pass

	def train(self, data, label, config, device, model_output_path):
		zipped_data = make_zipped_data(data, label)

		n_epoch = config.num_epochs        # Num of Epoch
		batch_size = config.batch_size      # 배치 사이즈
		save_step = 100 # 학습 저장 주기
		learning_rate = config.learning_rate

		self.model.to(device)

		dataset = KoElectraClassificationDataset(tokenizer=self.tokenizer, device=device, zipped_data=zipped_data, num_labels=config.num_of_classes, max_seq_len=config.max_len)
		train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
		{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
		'weight_decay': 0.01},
		{'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

		pre_epoch, pre_loss, train_step = 0, 0, 0
		if os.path.isfile(model_output_path):
			checkpoint = torch.load(model_output_path, map_location=device)
			pre_epoch = checkpoint['epoch']
			pre_loss = checkpoint['loss']
			train_step =  checkpoint['train_step']
			total_train_step =  checkpoint['total_train_step']

			self.model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

			print(f"load pretrain from: {model_output_path}, epoch={pre_epoch}, loss={pre_loss}")

		losses = []

		offset = pre_epoch
		for step in range(config.num_epochs):
			epoch = step + offset
			loss = self.train_model( config.num_epochs, self.model, optimizer, train_loader, save_step, model_output_path, train_step)
			losses.append(loss)

		# data
		data = {
			"loss": losses
		}
		df = pd.DataFrame(data)
		display(df)

		plt.figure(figsize=[12, 4])
		plt.plot(losses, label="loss")
		plt.legend()
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.show()

	def train_model(self, epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step = 0):
		losses = []
		train_start_index = train_step+1 if train_step != 0 else 0
		total_train_step = len(train_loader)
		self.model.train()

		with tqdm(total= total_train_step, desc=f"Train({epoch})") as pbar:
			pbar.update(train_step)
			for i, data in enumerate(train_loader, train_start_index):
				optimizer.zero_grad()
				inputs = {'input_ids': data['input_ids'],
					'attention_mask': data['attention_mask'],
					'labels': data['labels']
					}

				outputs = model(**inputs)

				loss = outputs[0]

				losses.append(loss.item())

				loss.backward()
				optimizer.step()

				pbar.update(1)
				pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")

				if i >= total_train_step or i % save_step == 0:
					torch.save({
						'epoch': epoch,  # 현재 학습 epoch
						'model_state_dict': model.state_dict(),  # 모델 저장
						'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
						'loss': loss.item(),  # Loss 저장
						'train_step': i,  # 현재 진행한 학습
						'total_train_step': len(train_loader)  # 현재 epoch에 학습 할 총 train step
					}, save_ckpt_path)

		return np.mean(losses)


class KoElectraClassificationDataset(Dataset):
	def __init__(self,
			device = 'cpu',
			tokenizer=None,
			zipped_data=None,
			num_labels=None,
			max_seq_len=None # KoElectra max_length
			):
		self.device = device
		self.data =[]
		self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()

		mock_datas = []

		for zd in zipped_data:
			d = []

			if len(zd[0]) > max_seq_len:
				d.append(zd[0][:self.max_seq_len])
			else:
				d.append(zd[0])

			d.append(zd[1])
			mock_datas.append(d)

		for mock_data in mock_datas:
			index_of_words = self.tokenizer.encode(mock_data[0])
			token_type_ids = [0] * len(index_of_words)
			attention_mask = [1] * len(index_of_words)

			# Padding Length
			padding_length = max_seq_len - len(index_of_words)

			# Zero Padding
			index_of_words += [0] * padding_length
			token_type_ids += [0] * padding_length
			attention_mask += [0] * padding_length

			# Label
			label = int(mock_data[1])
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

def make_zipped_data(data, label):		
	zipped_data = []

	for i in range(len(data)):
		row = []
		row.append(data[i])
		row.append(label[i])
		zipped_data.append(row)

		return zipped_data

