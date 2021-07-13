# charm-shin-han

## usage

### Data Maker

```python
from charm_shin_han.data_maker import DataMaker

train_shinhan_data, train_shinhan_label, test_shinhan_data, test_shinhan_label = DataMaker.make_shinhan_issue_class_data(
    "/content/drive/MyDrive/신한은행/training-data/Labeled_Data/shinhan_app_review_1.xlsx",
    "/content/drive/MyDrive/신한은행/training-data/text_data_index.txt",
    )

hana_data, hana_label = DataMaker.make_issue_class_data_from_crawled("/content/drive/MyDrive/신한은행/training-data/Labeled_Data/hana_app_review.xlsx")
woori_data, woori_label = DataMaker.make_issue_class_data_from_crawled("/content/drive/MyDrive/신한은행/training-data/Labeled_Data/woori_app_review.xlsx")
```

### KoBERT Classification Trainer

```python
from charm_shin_han.kobert_classification_trainer import KobertClassficationTrainer
from charm_shin_han.kobert_config import KoBERTConfig
import torch

train_data = [
  'hello', 'hi', 'im', 'shinhan', 'app review'
]
train_label = [0, 1, 2, 3, 4]
test_data = ['hi', 'hello']
test_label = [0, 1]

config = KoBERTConfig(
    num_of_classes=5,
    max_len = 256,
    batch_size = 64,
    warmup_ratio = 0.1,
    num_epochs = 5,
    max_grad_norm = 1,
    log_interval = 200,
    learning_rate =  5e-5,
)
device = torch.device("cuda:0")
trainer = KobertClassficationTrainer()
trainer.train(train_data, train_label, test_data, test_label, config, 'output.pt', device)
```

### KoElectra Classification Trainer

```python

```
