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



        return rank

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