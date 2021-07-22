import os
import argparse
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

        tokenizer.train(
            files=[self.corpus_file],
            limit_alphabet=self.limit_alphabet,
            vocab_size=self.vocab_size
        )

        tokenizer.save("./", "ch-{}-wpm-{}".format(self.limit_alphabet, self.vocab_size))    
    




# def load_vocab(vocab_file):
#     """Loads a vocabulary file into a dictionary."""
#     vocab = collections.OrderedDict()
#     index = 0
#     with open(vocab_file, "r", encoding="utf-8") as reader:
#         while True:
#             token = convert_to_unicode(reader.readline())
#             if not token:
#                 break
#             token = token.strip()
#             vocab[token] = index
#             index += 1
#     return vocab