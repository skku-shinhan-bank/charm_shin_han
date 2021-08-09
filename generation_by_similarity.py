from transformers import ElectraTokenizer
from numpy import dot
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm, tqdm_notebook

# Padding sequences by leng
def pad_seq(seq, leng):
    if len(seq) > leng:
        return seq[:leng]
    else:
        padding_len = leng - len(seq)
        return seq + [0] * padding_len

# Getting a cosine similarity
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

class GenerationBySimilarity:
    def __init__(self):
        return

    # Generates (actually, "finds") the answer by getting cosine similarity for each reviews
    def generate(self, input_data, review_data, answer_data, max_len):
        # use a tokenizer from KoELECTRA
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        tokenized_data = []

        for data in review_data:
            id_data = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data))
            id_data = pad_seq(id_data, max_len)
            tokenized_data.append(id_data)
        
        # make an integer encoding
        input_data = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_data))
        input_data = pad_seq(input_data, max_len)

        max_index = 0
        max_similarity = 0
        sec_index = 0
        sec_similarity = 0
        thi_index = 0
        thi_similarity = 0

        # Get most similar reviews and answers
        for index, review in enumerate(tqdm_notebook(tokenized_data)):
            temp = cos_sim(np.array(input_data), np.array(review))
            if (temp > max_similarity):
                thi_index = sec_index
                thi_similarity = sec_similarity
                sec_index = max_index
                sec_similarity = max_similarity
                max_similarity = temp
                max_index = index
            elif (temp > sec_similarity):
                thi_index = sec_index
                thi_similarity = sec_similarity
                sec_index = max_index
                sec_similarity = temp
            elif (temp > thi_similarity):
                thi_index = index
                thi_similarity = temp
            return answer_data[max_index], max_similarity