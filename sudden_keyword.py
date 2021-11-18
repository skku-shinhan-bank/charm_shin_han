import pandas as pd

class KeywordRank:
    def __init__(self):
        return

    def ranker(self, data):
        rank = []
        

        return rank

    def month_classifier(self, data_path):
        data = pd.read_excel(data_path)

        month_index = []
        month_data = []

        for i in range(len(data)):
            month_index.append(str(data['일자'][i][:7]))

        month_index = set(month_index)

        for m in month_index:
            temp = []
            for i in range(len(data)):
                if data['일자'][i][:7] == m:
                    temp.append(str(data['review'][i]))
            month_data.append(temp)

        return month_data