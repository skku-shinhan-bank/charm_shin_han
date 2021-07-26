import os
from tokenizers import BertWordPieceTokenizer
from konlpy.tag import Mecab

class WordpieceVocabTest :
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

        # mecab for window는 아래 코드 사용
        mecab_tokenizer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic").morphs
        print('mecab check :', mecab_tokenizer('어릴때보고 지금다시봐도 재밌어요ㅋㅋ'))

        for_generation = False # or normal

        if for_generation:
            # 1: '어릴때' -> '어릴, ##때' for generation model
            total_morph=[]
            for sentence in data:
                # 문장단위 mecab 적용
                morph_sentence= []
                count = 0
                for token_mecab in mecab_tokenizer(sentence):
                    token_mecab_save = token_mecab
                    if count > 0:
                        token_mecab_save = "##" + token_mecab_save  # 앞에 ##를 부친다
                        morph_sentence.append(token_mecab_save)
                    else:
                        morph_sentence.append(token_mecab_save)
                        count += 1
                # 문장단위 저장
                total_morph.append(morph_sentence)

        else:
            # 2: '어릴때' -> '어릴, 때'   for normal case
            total_morph=[]
            for sentence in data:
                # 문장단위 mecab 적용
                morph_sentence= mecab_tokenizer(sentence)
                # 문장단위 저장
                total_morph.append(morph_sentence)
                                
        # mecab 적용한 데이터 저장
        # ex) 1 line: '어릴 때 보 고 지금 다시 봐도 재밌 어요 ㅋㅋ'
        with open('after_mecab.txt', 'w', encoding='utf-8') as f:
            for line in total_morph:
                f.write(' '.join(line)+'\n')
