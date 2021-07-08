import pandas as pd
import numpy as np
import torch
import random
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from .model.koelectra_classifier import koElectraForSequenceClassification
from tqdm.notebook import tqdm
import torch
import os
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tqdm
from kobert_transformers import get_tokenizer
from transformers import (
  ElectraConfig,
  ElectraTokenizer
)

class KoelectraClassificationTrainer:
	def __init__(self, config):
		electra_config = ElectraConfig.from_pretrained("monologg/koelectra-small-v2-discriminator")
		model = koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path = "monologg/koelectra-small-v2-discriminator", config = electra_config, num_labels = config.num_label)
		tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

		self.tokenizer = tokenizer
		self.model = model
		pass

	def train(self, data, label, config, device, model_output_path):
		zippedData = []

		for i in range(len(data)):
			row = []

			row.append(data[i])
			row.append(label[i])

			zippedData.append(row)

			random.shuffle(zippedData)

		dataset_train = zippedData[:config.num_of_train_data]
		dataset_test = zippedData[config.num_of_train_data:]
		learning_rate = 5e-5

		self.model.to(device)

		dataset = WellnessTextClassificationDataset(tokenizer=self.tokenizer, device=device, zippedData=dataset_train, num_label = config.num_label, max_seq_len = config.max_seq_len)
		train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

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
		for step in range(config.n_epoch):
			epoch = step + offset
			loss = self.train_model( epoch, self.model, optimizer, train_loader, config.save_step, model_output_path, train_step)
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
	
		
	def train_model(epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step = 0):
		losses = []
		train_start_index = train_step+1 if train_step != 0 else 0
		total_train_step = len(train_loader)
		model.train()

		with tqdm(total= total_train_step, desc=f"Train({epoch})") as pbar:
			pbar.update(train_step)
			for i, data in enumerate(train_loader, train_start_index):
				optimizer.zero_grad()

				'''
				inputs = {'input_ids': batch[0],
						'attention_mask': batch[1],
						'bias_labels': batch[3],
						'hate_labels': batch[4]}
				if self.args.model_type != 'distilkobert':
				inputs['token_type_ids'] = batch[2]
				'''
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

	def get_model_and_tokenizer(self, device, save_ckpt_path):
		if os.path.isfile(save_ckpt_path):
			checkpoint = torch.load(save_ckpt_path, map_location=device)
			pre_epoch = checkpoint['epoch']
			self.model.load_state_dict(checkpoint['model_state_dict'])

			print(f"\n\nload pretrain from\n\n: {save_ckpt_path}, epoch={pre_epoch}")

		return self.model, self.tokenizer

	def get_model_input(data):
		return {'input_ids': data['input_ids'],
					'attention_mask': data['attention_mask'],
					'labels': data['labels']
					}

	def evaluate(self, device, batch_size, dataset_test, num_label, max_seq_len, save_ckpt_path):

		model, tokenizer = self.get_model_and_tokenizer(device, save_ckpt_path)
		model.to(device)

		eval_dataset = self.WellnessTextClassificationDataset(device=device, tokenizer=tokenizer, zippedData=dataset_test, num_label = num_label, max_seq_len = max_seq_len)
		eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

		loss = 0
		acc = 0

		model.eval()
		for data in tqdm(eval_dataloader, desc="Evaluating"):
			with torch.no_grad():
				inputs = self.get_model_input(data)
				outputs = model(**inputs)
				loss += outputs[0]
				logit = outputs[1]
				acc += (logit.argmax(1)==inputs['labels']).sum().item()
				print('\n\n가나다라마바사', logit.argmax(1))

		return loss / len(eval_dataset), acc / len(eval_dataset)

	def evaluate_koelectra(self, dataset_test, batch_size, num_label, max_seq_len, save_ckpt_path):
		# n_epoch = 5  # Num of Epoch
		# batch_size = 1  # 배치 사이즈
		ctx = "cuda" if torch.cuda.is_available() else "cpu"
		device = torch.device(ctx)
		eval_loss, eval_acc = self.evaluate(device, batch_size, dataset_test, num_label, max_seq_len, save_ckpt_path)
		print(f'\tLoss: {eval_loss:.4f}(valid)\t|\tAcc: {eval_acc * 100:.1f}%(valid)')


class WellnessTextClassificationDataset(Dataset):
  def __init__(self,
               device = 'cpu',
               tokenizer = None,
               zippedData = None,
        	   num_label = None,
               max_seq_len = None, # KoBERT max_length
               ):

    self.device = device
    self.data =[]
    self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()

    mock_datas = []

    for zd in zippedData:
      d = []

      if len(zd[0]) > max_seq_len:
        d.append(zd[0][:512])
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