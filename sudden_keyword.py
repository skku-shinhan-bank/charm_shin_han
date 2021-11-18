import pandas as pd
import numpy as np
from .keyword_extracter import KeywordExtracter

class SuddenKeyword:
    def __init__(self):
        self.ke = KeywordExtracter()
        return

    def make_top10(self, data_path):
        top10 = []

        data = pd.read_excel(data_path)
        month, month_index = self.month_classifier(data)
        
        for i in month_index:
            top10.append(self.ke.analyze(list(month[i])))


        return top10

    def month_classifier(self, data):
            month_index = []
            month_data = []

            for i in range(len(data)):
                month_index.append(str(data['일자'][i])[:7])

            month_index = set(month_index)

            for m in month_index:
                temp = []
                for i in range(len(data)):
                    if str(data['일자'][i])[:7] == m:
                        temp.append(data['review'][i])
                month_data.append(temp)

            month = pd.DataFrame(month_data)
            n = 0
            for i in month_index:
                month = month.rename(columns={n:i})
                n += 1

            return month, month_index