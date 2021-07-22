import os
import argparse
from tokenizers import BertWordPieceTokenizer

class WordpieceVocab :
    def __init__(self):
        pass
    def vocab(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--corpus_file", type=str)
        parser.add_argument("--vocab_size", type=int, default=32000)
        parser.add_argument("--limit_alphabet", type=int, default=6000)

        args = parser.parse_args()

        tokenizer = BertWordPieceTokenizer(
            vocab_file=None,
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False, # Must be False if cased model
            lowercase=False,
            wordpieces_prefix="##"
        )

        tokenizer.train(
            files=[args.corpus_file],
            limit_alphabet=args.limit_alphabet,
            vocab_size=args.vocab_size
        )

        tokenizer.save("./", "ch-{}-wpm-{}".format(args.limit_alphabet, args.vocab_size))    
    




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