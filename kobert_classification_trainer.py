from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import random
import torch
import gluonnlp as nlp
from .model.kobert_classifier import KoBERTClassifier
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from torch import nn
from tqdm import tqdm, tqdm_notebook
import numpy as np
from torch.utils.data import Dataset
import time
from .confusion_matrix import ConfusionMatrix
import pandas as pd

class KobertClassficationTrainer:
  def __init__(self):
    return

  def train(self, train_data, train_label, test_data, test_label, config, model_output_path, device):
    bert_model, vocab = get_pytorch_kobert_model()
    tok = get_tokenizer()
    tokenizer = nlp.data.BERTSPTokenizer(tok, vocab, lower=False)

    dataset_train = []
    dataset_test = []

    for i in range(len(train_data)):
        row = []

        row.append(train_data[i])
        row.append(train_label[i])

        dataset_train.append(row)

    for i in range(len(test_data)):
        row = []

        row.append(test_data[i])
        row.append(test_label[i])

        dataset_test.append(row)

    random.shuffle(dataset_train)
    random.shuffle(dataset_test)

    data_train = KoBERTDataset(dataset_train, 0, 1, tokenizer, config.max_len, True, False)
    data_test = KoBERTDataset(dataset_test, 0, 1, tokenizer, config.max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=config.batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=config.batch_size, num_workers=5)
    
    classification_model = KoBERTClassifier(bert_model,  dr_rate=0.5, num_classes=config.num_of_classes).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {
        'params': [p for n, p in classification_model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
      },
      {
        'params': [p for n, p in classification_model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,
      }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    t_total = len(train_dataloader) * config.num_epochs
    warmup_step = int(t_total * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    for epoch_index in range(config.num_epochs):
      print("[epoch {}]\n".format(epoch_index + 1))
      train_acc = 0.0
      test_acc = 0.0
      train_losses = []
      start_time = time.time()
      classification_model.train()
      for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = classification_model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        train_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classification_model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        # if batch_id % config.log_interval == 0:
        #   print("batch id {} / loss {} / train acc {}".format(batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
      train_loss = np.mean(train_losses)
      print("train acc {} / loss {} / train time {}".format(train_acc / (batch_id+1), train_loss,time.time() - start_time))

      cm = ConfusionMatrix(config.num_of_classes)
      classification_model.eval()
      for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = classification_model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)

        for index, real_class_id in enumerate(label):
          max_vals, max_indices = torch.max(out, 1)
          cm.add(real_class_id, max_indices[index].item())
      print("test acc {}".format(test_acc / (batch_id+1)))
      print("<confusion matrix>\n", pd.DataFrame(cm.get()))
      print('\n')

    torch.save(classification_model.state_dict(), model_output_path)

class KoBERTDataset(Dataset):
  def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
              pad, pair):
    transform = nlp.data.BERTSentenceTransform(
        bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

    self.sentences = [transform([i[sent_idx]]) for i in dataset]
    self.labels = [np.int32(i[label_idx]) for i in dataset]

  def __getitem__(self, i):
    return (self.sentences[i] + (self.labels[i], ))

  def __len__(self):
    return (len(self.labels))


def calc_accuracy(X,Y):
  max_vals, max_indices = torch.max(X, 1)
  train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
  return train_acc
  