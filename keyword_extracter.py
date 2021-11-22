from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
import numpy as np
#import torch
from tqdm import notebook
import mecab
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordExtracter:
  def __init__(self):
    mecab_ = mecab.MeCab()
    # POS tagger
    self.pos = mecab_.pos
    # self.corpus_list: contains words (that can be keywords) for each reviews
    # self.keyword_set: every keyword in the data
    # self.keyword_rank: ranking of keyword
    # self.tfidf_matrix: contains tf-idf value of each word in the self.corpus_list
    self.keyword_tf = defaultdict(int)  # Term Frequency value of keyword

  def to_surface(self, tok1, tok2, tok3=''):
    if (tok1+tok2+tok3 in self.synonym_dict):
      return self.synonym_dict[tok1+tok2+tok3]
    return tok1 + tok2 + tok3

  # 문장 별로 단축된 tfidf를 단어와 함께 반환
  def get_short_tfidf(self, NUM_SENTENCE):
    tfidf_sent = [str(value) for value in self.tfidfv_matrix[NUM_SENTENCE] if not value == 0]
    #print(shinhan_data[NUM_SENTENCE])
    short_dict = {}
    for key, value in self.tfidfv.vocabulary_.items():
      if self.tfidfv_matrix[NUM_SENTENCE][value] != 0:
        #print(key, "(",tfidfv_matrix[NUM_SENTENCE][value],"),", end=" ")
        short_dict[key] = self.tfidfv_matrix[NUM_SENTENCE][value]

    short_dict = sorted(short_dict.items(), reverse=True, key = lambda item: item[1]) #sorting
    return short_dict

  def analyze(self, data, ngram_threshold = 5, pmi_threshold = 0.0001, sim_threshold = 10, keyword_threshold = 3,
              use_noun = True, use_predicate = True, synonym_dict = {}, stopword = []): # return self.keyword_rank as List[(keyword as str, frequency as int), ....]
    # data: (list type) review to be analyzed
    # ngram_threshold: The minimum value of the number of words to be registered as n-gram
    # pmi_threshold: The minimum value of the PMI value of n-gram to be registered as keyword
    # keyword_threshold: The minumum value of the number of terms to be in the corpus list
    # use_noun: make nouns can be keywords
    # use_predicate: make predicates can be keywords
    # synonym_dict: (dictionary type) convert words to their synonym
    self.corpus_list = []
    self.synonym_dict = synonym_dict

    monogram_list = [] # for monogram
    ngram_list = [] # for bi-gram and tri-gram

    append_pos_list = [] # only words whose POS in the list can be the keyword
    noun_list = ['NNG', 'NNP', 'NF']
    predicate_list = ['VV', 'VA', 'MM']
    if use_noun:
      append_pos_list += noun_list
    if use_predicate:
      append_pos_list += predicate_list
    josa_list = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC'] # JOSA

    print("Collecting n-grams...")
    for sent in notebook.tqdm(data):
      monogram_list.append([])
      ngram_list.append([])
      pos_sent = self.pos(sent)
      temp_corpus = []
      ngram_corpus = []
      for i, word in enumerate(pos_sent):
        temp_corpus.append(word[0]) # append monogram regardless of POS
        if word[1] in append_pos_list:
          if i < len(pos_sent) - 1:
            if pos_sent[i+1][1] in josa_list: # 조사가 나온 경우 n-gram에서 빼기 위함
              continue
            ngram_corpus.append(tuple([word[0], pos_sent[i+1][0]])) # tuple을 써야 hashing이 가능하기에 n-gram은 tuple로 추가
          if i < len(pos_sent) - 2:
            if pos_sent[i+2][1] in josa_list:
              continue
            ngram_corpus.append(tuple([word[0], pos_sent[i+1][0], pos_sent[i+2][0]]))
      monogram_list.append(temp_corpus)
      ngram_list.append(ngram_corpus)

    # 정리된 monogram과 n-gram 내에서 각각의 빈도값을 추려본다
    monogram_counter = defaultdict(int)
    ngram_counter = defaultdict(int)
    for sent in monogram_list:
      for monogram in sent:
        monogram_counter[monogram] += 1
    for sent in ngram_list:
      for ngram in sent:
        ngram_counter[ngram] += 1

    # PMI: (#ngram - threhold)/∏#monogram임을 이용하여 PMI 값을 구함
    ngram_score = defaultdict(float)
    self.keyword_set = set()
    for key in ngram_counter:
      if len(key) == 2:
        ngram_score[key] = float( (ngram_counter[key] - ngram_threshold) / (monogram_counter[key[0]] * monogram_counter[key[1]]) )
      else:
        ngram_score[key] = float( (ngram_counter[key] - ngram_threshold) / (monogram_counter[key[0]] * monogram_counter[key[1]] * monogram_counter[key[2]]) )
    print("Get n-gram keyword by PMI")
    for sent in notebook.tqdm(data):
      pos_sent = self.pos(sent)
      temp_corpus = []
      for i, word in enumerate(pos_sent):
        if word[1] in append_pos_list:
          if i < len(pos_sent) - 1: # bi-gram과 tri-gram 간의 PMI 값을 비교한 뒤 더 높은 쪽을 단어로 추가함. PMI threshold를 넘지 못했다면 monogram을 대신 추가하고 넘김
            if i < len(pos_sent) - 2:
              bigram_score = ngram_score[(word[0], pos_sent[i+1][0])]
              trigram_score = ngram_score[(word[0], pos_sent[i+1][0], pos_sent[i+2][0])]
              if bigram_score > trigram_score and bigram_score > pmi_threshold:
                bigram = self.to_surface(word[0], pos_sent[i+1][0])
                temp_corpus.append(bigram)
                self.keyword_set.add(bigram)
                self.keyword_tf[bigram] += 1
                i+=1
                continue
              if trigram_score > bigram_score and trigram_score > pmi_threshold:
                trigram = self.to_surface(word[0], pos_sent[i+1][0], pos_sent[i+2][0])
                temp_corpus.append(trigram)
                self.keyword_set.add(trigram)
                self.keyword_tf[trigram] += 1
                i+=2
                continue
            else:
              if ngram_score[(word[0], pos_sent[i+1][0])] > pmi_threshold:
                bigram = self.to_surface(word[0], pos_sent[i+1][0])
                temp_corpus.append(bigram)
                self.keyword_set.add(bigram)
                self.keyword_tf[bigram] += 1
                i+=1
                continue
          if word[0] in synonym_dict:
            temp_corpus.append(synonym_dict[word[0]])
            self.keyword_set.add(synonym_dict[word[0]])
            self.keyword_tf[synonym_dict[word[0]]] += 1
          else:
            temp_corpus.append(word[0])
            self.keyword_set.add(word[0])
            self.keyword_tf[word[0]] += 1
          
      self.corpus_list.append(temp_corpus)
    
    for rev in self.corpus_list:
      for term in rev:
        self.keyword_tf[term] += 1
    
    for rev in self.corpus_list:
      for term in rev:
        if self.keyword_tf[term] < keyword_threshold: # if the keyword frequency is less then keyword_threshold, delete it from the corpus list.
          rev.remove(term)
          continue
        if term in stopword:
          rev.remove(term)

    print("TF-IDF")
    self.tfidfv = TfidfVectorizer(preprocessor = ' '.join)
    self.tfidf_matrix = self.tfidfv.fit_transform(self.corpus_list).toarray()
    #self.dic_list = []
    
    self.keyword_rank = {}
    tfidf_voc = sorted(self.tfidfv.vocabulary_.items())

    for i in notebook.tqdm(self.tfidf_matrix):
      kw = tfidf_voc[i.argmax()][0]
      if kw in self.keyword_rank:
        self.keyword_rank[kw] += 1
      else:
        self.keyword_rank[kw] = 1

    self.keyword_rank = sorted(self.keyword_rank.items(), reverse=True, key = lambda item: item[1]) #sorting
    # Extracting similar keyword list!
    self.get_similar_keyword_list(sim_threshold = sim_threshold) # get similar keyword list
    return self.keyword_rank

  def get_similar_keyword_list(self, sim_threshold = 10):
    self.similar_keyword = {}
    if not self.corpus_list: # if corpus_list is empty
      print("Analyze the data first!")
      return
    for rev in notebook.tqdm(self.corpus_list):
      for term in rev:
        if term not in self.similar_keyword:
          self.similar_keyword[term] = {}
      for i, term1 in enumerate(rev):
        for term2 in rev[i+1:]:
          if term1 == term2:
            continue # avoid duplication problem
          if term2 not in self.similar_keyword[term1]:
            self.similar_keyword[term1][term2] = 0
          self.similar_keyword[term1][term2] += 1
          if term1 not in self.similar_keyword[term2]:
            self.similar_keyword[term2][term1] = 0
          self.similar_keyword[term2][term1] += 1
    for term1 in self.similar_keyword:
      for term2 in self.similar_keyword[term1]:
        self.similar_keyword[term1][term2] -= sim_threshold
        self.similar_keyword[term1][term2] /= self.keyword_tf[term1] * self.keyword_tf[term2]
  def get_similar_keyword(self, keyword):
    if keyword not in self.similar_keyword:
      return
    return sorted(self.similar_keyword[keyword].items(), reverse=True, key = lambda item: item[1])
  def view_review(self, data, review_idx):
    print(data[review_idx])
    print(self.dic_list[review_idx])