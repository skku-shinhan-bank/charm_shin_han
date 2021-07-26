import os
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer
from .model.koelectra_classifier import KoElectraClassifier
from transformers import (
	ElectraConfig,
)


class WordpieceVocabTest :
    def __init__(self):
        pass

    def make_vocab(self, corpus_file, vocab_size, limit_alphabet):
        
        tokenizer = BertWordPieceTokenizer(
            vocab=None,
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False, # Must be False if cased model
            lowercase=False,
            wordpieces_prefix="##"
        )

        tokenizer.train(
            files=[corpus_file],
            vocab_size=vocab_size,
            limit_alphabet=limit_alphabet
        )

        checkpoint_path ="checkpoint"
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        tokenizer.save_model("./checkpoint")

		# electra_config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
		# classification_model = KoElectraClassifier.from_pretrained(pretrained_model_name_or_path = "monologg/koelectra-base-v3-discriminator", config = electra_config, num_labels = config.num_label)
		# new_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

        origin_tokens = []
        f = open("/content/checkpoint/vocab.txt", 'r')
        while True :
            line = f.readline()
            if not line :
                break
        origin_tokens.append(line)
        f.close()
        new_tokens=origin_tokens[5:]
        print(new_tokens)
		# new_tokenizer.add_tokens(new_tokens)
		# classification_model.resize_token_embeddings(len(tokenizer))

    # def retrain_tokenizer(self, config):

	# 	electra_config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
	# 	classification_model = KoElectraClassifier.from_pretrained(pretrained_model_name_or_path = "monologg/koelectra-base-v3-discriminator", config = electra_config, num_labels = config.num_label)
    #     new_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    #     origin_tokens = []
    #     f = open("/checkpoint/vocab.txt", 'r')
    #     while True :
    #         line = f.readline()
    #         if not line :
    #             break
    #         origin_tokens.append(line)
    #     f.close()
    #     new_tokens=origin_tokens[5:]
    #     new_tokenizer.add_tokens(new_tokens)

