from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
import gc
from transformers import (
	ElectraConfig
)
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from .model.koelectra_classifier import KoElectraClassifier
from .koelectra_classification_trainer import KoElectraClassificationDataset

class SimilarityComparator:
  def __init__(self):
    pass

  def mean_pooling(self, model_output, attention_mask):
    #Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

  def compare_generate_review(self, comparator_model_path, str1, data, comment):

    data.insert(0, self.string)

    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = AutoModel.from_pretrained(comparator_model_path).to("cuda")

    #Tokenize sentences
    encoded_input = tokenizer(data, padding=True, truncation=True, max_length=32, return_tensors='pt')

    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
      model_output = model(input_ids=encoded_input["input_ids"].to("cuda"))

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'].to("cuda"))
    cosine_scores = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

    org = 0
    temp = cosine_scores[org]
    temp.argsort(descending=True)

    for i in temp.argsort(descending=True)[0:10]:
      print(f"{i}. {data[org]} <> {data[i]} \nScore: {cosine_scores[org][i]:.4f}")

    for i in temp.argsort(descending=True)[0:10]:
        print("review")
        print(data[i])
        print("comment")
        print(comment[i])

