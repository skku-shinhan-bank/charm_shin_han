import os
from tokenizers import BertWordPieceTokenizer
from konlpy.tag import Mecab

#Mecab 사용

class WordpieceVocabMecab :
    def __init__(self):
        pass

    def make_vocab(self, file_path, vocab_size, limit_alphabet):
        self.generate_mecab_vocab(file_path)

        tokenizer = BertWordPieceTokenizer(
            vocab=None,
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False, # Must be False if cased model
            lowercase=False,
            wordpieces_prefix="##"
        )

        tokenizer.train(
            files=['after_mecab.txt'],
            vocab_size=vocab_size,
            limit_alphabet=limit_alphabet,
            min_frequency = 5,
            show_progress=True
        )

        checkpoint_path ="checkpoint"
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        tokenizer.save_model("./checkpoint")

    def generate_mecab_vocab(self, file_path):

        data = open(file_path, 'r').read().split('\n')

        mecab_tokenizer = Mecab()
        for_generation = False # or normal

        if for_generation:
            total_morph=[]
            for sentence in data:
                morph_sentence= []
                count = 0
                for token_mecab in mecab_tokenizer.morphs(sentence):
                    token_mecab_save = token_mecab
                    if count > 0:
                        token_mecab_save = "##" + token_mecab_save  # 앞에 ##를 부친다
                        morph_sentence.append(token_mecab_save)
                    else:
                        morph_sentence.append(token_mecab_save)
                        count += 1
                total_morph.append(morph_sentence)
        else:
            total_morph=[]
            for sentence in data:
                morph_sentence= mecab_tokenizer.morphs(sentence)
                total_morph.append(morph_sentence)
                                
        with open('after_mecab.txt', 'w', encoding='utf-8') as f:
            for line in total_morph:
                f.write(' '.join(line)+'\n')
