import pandas as pd
import numpy as np
# from .keyword_extracter import KeywordExtracter

class SuddenKeyword:
    def __init__(self):
        # self.ke = KeywordExtracter()
        return

    def make_top10(self, month, index):
        rank = pd.DataFrame()

        # for m in index:
        #     list1 = pd.DataFrame(month, columns=[m])
        #     list1 = list1.dropna()
        #     list__ = list1.values.tolist()
        #     list = []
        #     for i in list__:
        #         list.append(*i)
        #     temp = self.ke.analyze(list)
        #     temp = pd.DataFrame(temp)
        #     temp = temp.rename(columns={0:m})
        #     rank = pd.concat([rank, temp], axis=1)

        keyword_rank = self.extract_sudden_keyword(rank, index)


        return keyword_rank

    def month_classifier(self, data_path):
        data = pd.read_excel(data_path)

        month_index = []

        for i in range(len(data)):
            month_index.append(str(data['일자'][i])[:7])

        month_index = set(month_index)
        month_index = list(month_index)
        month_index.sort()

        month = []
        for i in range(len(data)):
            if str(data['일자'][i])[:7] == month_index[0]:
                month.append(data['review'][i])

        month = pd.DataFrame()
        for m in month_index:
            temp = []
            for i in range(len(data)):
                if str(data['일자'][i])[:7] == m:
                    temp.append(data['review'][i])
            temp = pd.DataFrame(temp)
            temp = temp.rename(columns={0:m})
            month = pd.concat([month, temp], axis=1)

        return month, month_index
    
    def extract_sudden_keyword(self, rank, index):
        keyword_rank = pd.DataFrame()

        for m in range(len(index)-1):
            pre_keyword_rank = []
            for i in range(3):
                keyword_c = rank[m][i]  #수
                keyword_d = rank[index[m]][i]  #키워드

                try:
                    a = rank.index[(rank[index[m-1]] == keyword_d)].tolist()[0]
                except:
                    count_a = 0
                else:
                    count_a = rank[m-1][a]

                try:
                    b = rank.index[(rank[index[m+1]] == keyword_d)].tolist()[0]
                except:
                    count_b = 0
                else:
                    count_b = rank[m+1][b]

                if keyword_c > count_a and keyword_c > count_b:
                    pre_keyword_rank.append(keyword_d)
                else:
                    continue
            pre_keyword_rank = pd.DataFrame(pre_keyword_rank)
            pre_keyword_rank = pre_keyword_rank.rename(columns={0:index[m]})
            keyword_rank = pd.concat([keyword_rank, pre_keyword_rank], axis=1)

        return keyword_rank
