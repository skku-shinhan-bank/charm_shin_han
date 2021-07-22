import os
from tokenizers import BertWordPieceTokenizer

class WordpieceVocab :
    def __init__(self, corpus_file, vocab_size, limit_alphabet):
        self.corpus_file = corpus_file
        self.vocab_size = vocab_size
        self.limit_alphabet = limit_alphabet

        pass

    def vocab(self):
        tokenizer = BertWordPieceTokenizer(
            vocab_file=None,
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False, # Must be False if cased model
            lowercase=False,
            wordpieces_prefix="##"
        )
        # tokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

        tokenizer.train(
            files=[self.corpus_file],
            limit_alphabet=self.limit_alphabet,
            vocab_size=self.vocab_size
        )

        tokenizer.save("./", "ch-{}-wpm-{}".format(args.limit_alphabet, args.vocab_size))
        