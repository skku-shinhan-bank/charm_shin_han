import argparse
import os
from tokenizers import BertWordPieceTokenizer

class KoElectra_vocab:
    def __init__(self):
        pass

    def Wordpeiece(corpus_file, limit_alphabet, vocab_size):
        
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
            limit_alphabet=limit_alphabet,
            vocab_size=vocab_size
        )

        checkpoint_path ="vocab"
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        tokenizer.save_model("./vocab")