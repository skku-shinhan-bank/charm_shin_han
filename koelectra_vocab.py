import argparse
from tokenizers import BertWordPieceTokenizer

class KoElectra_vocab:
    def __init__():
        pass

    def Wordpeiece(self, corpus_file, limit_alphabet, vocab_size):
        
        self.corpus_file = corpus_file, 
        self.limit_alphabet = limit_alphabet, 
        self.vocab_size = vocab_size,
        
        # parser = argparse.ArgumentParser()

        # parser.add_argument(corpus_file)
        # parser.add_argument("--vocab_size", type=int, default=22000) # 만들 Vocab의 숫자 
        # parser.add_argument("--limit_alphabet", type=int, default=6000)

        # args = parser.parse_args()

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

        tokenizer.save("./ch-{}-wpm-{}-pretty".format(self.limit_alphabet, self.vocab_size),True)