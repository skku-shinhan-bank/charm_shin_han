import os
from tokenizers import BertWordPieceTokenizer

class WordpieceVocab :
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
