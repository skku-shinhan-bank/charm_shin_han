import torch
import os
from tqdm.notebook import tqdm
from tqdm import tqdm
from .model.koelectra_classifier import koElectraForSequenceClassifier
from .koelectra_classifcation_trainer import WellnessTextClassificationDataset
from .koelectra_classifcation_trainer import KoElectraClassficationTrainer
from transformers import (
  ElectraTokenizer,  
  ElectraConfig
)


class KoElectraClassficationEvaluator:
    def __init__(self):
        pass
    
    def get_model_and_tokenizer(self, device, config):
        # save_ckpt_path = CHECK_POINT[model_name]
        save_ckpt_path = 'checkpoint/config.model_output_path'

        # # if model_name== "koelectra":
        # model_name_or_path = "monologg/koelectra-small-v2-discriminator"

        # tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)
        # electra_config = ElectraConfig.from_pretrained(model_name_or_path)
        # model = koElectraForSequenceClassifier.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=electra_config, num_labels=config.num_of_classes)

        if os.path.isfile(save_ckpt_path):
            checkpoint = torch.load(save_ckpt_path, map_location=device)
            pre_epoch = checkpoint['epoch']
            # pre_loss = checkpoint['loss']
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"\n\nload pretrain from\n\n: {save_ckpt_path}, epoch={pre_epoch}")

        return model, tokenizer

    def get_model_input(self, data):
        return {'input_ids': data['input_ids'],
                    'attention_mask': data['attention_mask'],
                    'labels': data['labels']
                    }

    def evaluate(self, device, test_datas, config):

        model, tokenizer = self.get_model_and_tokenizer(device, config)
        model.to(device)

        # WellnessTextClassificationDataset 데이터 로더
        eval_dataset = WellnessTextClassificationDataset(device=device, tokenizer=tokenizer, zippedData=test_datas, num_labels=config.num_of_classes, max_seq_len=config.max_len)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.batch_size)

        # logger.info("***** Running evaluation on %s dataset *****")
        # logger.info("  Num examples = %d", len(eval_dataset))
        # logger.info("  Batch size = %d", batch_size)

        loss = 0
        acc = 0

        model.eval()
        for data in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                inputs = self.get_model_input(data)
                outputs = self.model(**inputs)
                loss += outputs[0]
                logit = outputs[1]
                acc += (logit.argmax(1)==inputs['labels']).sum().item()
                print('\n\n가나다라마바사', logit.argmax(1))
                # predict = logit.argmax()
                # num=predict.item()
                # print("카테고리값 : ",num)

        return loss / len(eval_dataset), acc / len(eval_dataset)

    def evaluate_koelectra(self, test_datas, config):
        n_epoch = config.num_epochs  # Num of Epoch
        batch_size = config.batch_size  # 배치 사이즈
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(ctx)
        # model_names=["kobert","koelectra"]
        # for model_name in model_names:
        eval_loss, eval_acc = self.evaluate(device, batch_size, test_datas)
        print(f'\tLoss: {eval_loss:.4f}(valid)\t|\tAcc: {eval_acc * 100:.1f}%(valid)')