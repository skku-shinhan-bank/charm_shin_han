from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import random
import torch
import gluonnlp as nlp
from kobert_dataset import KoBERTDataset
from kobert_classifier import KoBERTClassifier
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from torch import nn
from tqdm import tqdm, tqdm_notebook

class KobertClassficationTrainer:
  def __init__(self):
    pass

  def calc_accuracy(self,X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

  def train(self, reviewData, labelData, config):
    device = torch.device("cuda:0")

    bertmodel, vocab = get_pytorch_kobert_model()

    zippedData = []

    for i in range(len(reviewData)):
        row = []

        row.append(reviewData[i])
        row.append(labelData[i])

        zippedData.append(row)

    random.shuffle(zippedData)

    dataset_train = zippedData[:config.num_of_train_data]
    dataset_test = zippedData[config.num_of_train_data:]

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 10
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5

    data_train = KoBERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = KoBERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
    
    model = KoBERTClassifier(bertmodel,  dr_rate=0.5, num_classes=config.num_of_classes).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += self.calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += self.calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))

    return model