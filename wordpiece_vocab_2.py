import os
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer

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
            limit_alphabet=limit_alphabet,
            min_frequency = 5,
            show_progress=True
        )

        checkpoint_path ="checkpoint"
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        tokenizer.save_model("./checkpoint")

        user_defined_symbols = ['[BOS]','[EOS]','[UNK0]','[UNK1]','[UNK2]','[UNK3]','[UNK4]','[UNK5]','[UNK6]','[UNK7]','[UNK8]','[UNK9]']
        unused_token_num = 200
        unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
        user_defined_symbols = user_defined_symbols + unused_list

        new_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        origin_tokens = open('/content/checkpoint/vocab.txt', 'r').read().split('\n')
        new_tokens=origin_tokens[5:]
        new_tokenizer.add_tokens(new_tokens)

        new_tokenizer.get_vocab()
        new_tokenizer.all_special_tokens()
        special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
        new_tokenizer.add_special_tokens(special_tokens_dict)

        if not os.path.isdir(checkpoint_special):
            os.mkdir(checkpoint_special)
        new_tokenizer.save_model("./checkpoint_special")
