import torch
import os
from tqdm.notebook import tqdm
from tqdm import tqdm
from .model.koelectra_classifier import KoElectraClassifier
from .koelectra_classifcation_trainer import KoElectraClassificationDataset, make_zipped_data
from transformers import (
  ElectraTokenizer,  
  ElectraConfig
)


class KoElectraClassficationEvaluator:
    def __init__(self, device, model_path, config):
        save_ckpt_path = model_path
        model_name_or_path = "monologg/koelectra-small-v2-discriminator"

        tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)
        electra_config = ElectraConfig.from_pretrained(model_name_or_path)
        model = KoElectraClassifier.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=electra_config, num_labels=config.num_of_classes)

        if os.path.isfile(save_ckpt_path):
            checkpoint = torch.load(save_ckpt_path, map_location=device)
            pre_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"\n\nload pretrain from\n\n: {save_ckpt_path}, epoch={pre_epoch}")
        
        self.model = model
        self.tokenizer = tokenizer

        pass

    def get_model_input(self, data):
        return {'input_ids': data['input_ids'],
                'attention_mask': data['attention_mask'],
                'labels': data['labels']
                }

    def evaluate_model(self, test_datas, device, config, model_path):

        self.model.to(device)

        # KoElectraClassificationDataset 데이터 로더
        eval_dataset = KoElectraClassificationDataset(device=device, tokenizer=self.tokenizer, zipped_data=test_datas, num_labels=config.num_of_classes, max_seq_len=config.max_len)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.batch_size)

        loss = 0
        acc = 0

        self.model.eval()
        for data in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                inputs = self.get_model_input(data)
                outputs = self.model(**inputs)
                loss += outputs[0]
                logit = outputs[1]
                acc += (logit.argmax(1)==inputs['labels']).sum().item()
                print('\n\n최댓값', logit.argmax(1))

        return loss / len(eval_dataset), acc / len(eval_dataset)

    def evaluate(self, data, label, config, device, model_path):
        test_datas = make_zipped_data(data, label)
        eval_loss, eval_acc = self.evaluate_model(test_datas, device, config, model_path)
        print(f'\tLoss: {eval_loss:.4f}(valid)\t|\tAcc: {eval_acc * 100:.1f}%(valid)')

