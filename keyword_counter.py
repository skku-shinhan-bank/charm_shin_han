import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class KeywordCouter:
    def __init__(self):
        return
    
    def makecount(self, issue_0, issue_1, issue_2, issue_3, issue_4, issue_5):  #list
        issue0 = self.makeDataFrame(issue_0)
        issue1 = self.makeDataFrame(issue_1)
        issue2 = self.makeDataFrame(issue_2)
        issue3 = self.makeDataFrame(issue_3)
        issue4 = self.makeDataFrame(issue_4)
        issue5 = self.makeDataFrame(issue_5)

        pre_top5 = []
        for i in range(5):
            pre_top5.append(issue0[0][i])
            pre_top5.append(issue1[0][i])
            pre_top5.append(issue2[0][i])
            pre_top5.append(issue3[0][i])
            pre_top5.append(issue4[0][i])
            pre_top5.append(issue5[0][i])

        top = set(pre_top5)
        top5 = list(top)    

        issue00 = self.counter(issue0, top5)
        issue11 = self.counter(issue1, top5)
        issue22 = self.counter(issue2, top5)
        issue33 = self.counter(issue3, top5)
        issue44 = self.counter(issue4, top5)
        issue55 = self.counter(issue5, top5)

        pre_count = []
        pre_count.append(issue00)
        pre_count.append(issue11)
        pre_count.append(issue22)
        pre_count.append(issue33)
        pre_count.append(issue44)
        pre_count.append(issue55)
        pre_count = np.transpose(pre_count)
        count = pd.DataFrame.from_records(pre_count)
        count = count.rename(columns={0:'실행기능'})
        count = count.rename(columns={1:'로그인'})
        count = count.rename(columns={2:'회원가입'})
        count = count.rename(columns={3:'금융'})
        count = count.rename(columns={4:'기타'})
        count = count.rename(columns={5:'앱외부'})
        count = count.set_index(keys=[top5], inplace=False)

        count.plot()

        return count

    def issueClassifier(self, data_path):
        origin_data = pd.read_excel(data_path)
        issue_0_data = []
        issue_1_data = []
        issue_2_data = []
        issue_3_data = []
        issue_4_data = []
        issue_5_data = []

        #issue_0 실행기능
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 2 or origin_data['issue-function'][i] == 43):
                issue_0_data.append(str(origin_data['review'][i]))
        
        #issue_1 로그인
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 0 or origin_data['issue-function'][i] == 1 or origin_data['issue-function'][i] == 12
            or origin_data['issue-function'][i] == 52or origin_data['issue-function'][i] == 16):
                issue_1_data.append(str(origin_data['review'][i]))

         #issue_2 회원가입
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 3 or origin_data['issue-function'][i] == 7):
                issue_2_data.append(str(origin_data['review'][i]))

        #issue_3 긍융
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 4 or origin_data['issue-function'][i] == 10 or origin_data['issue-function'][i] == 33
            or origin_data['issue-function'][i] == 33 or origin_data['issue-function'][i] == 34 or origin_data['issue-function'][i] == 35
            or origin_data['issue-function'][i] == 36 or origin_data['issue-function'][i] == 37 or origin_data['issue-function'][i] == 15
            or origin_data['issue-function'][i] == 50 or origin_data['issue-function'][i] == 46 or origin_data['issue-function'][i] == 56
            or origin_data['issue-function'][i] == 58 or origin_data['issue-function'][i] == 18 or origin_data['issue-function'][i] == 63
            or origin_data['issue-function'][i] == 66):
                issue_3_data.append(str(origin_data['review'][i]))

        #issue_4 기타
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 13 or origin_data['issue-function'][i] == 14 or origin_data['issue-function'][i] == 26
            or origin_data['issue-function'][i] == 8 or origin_data['issue-function'][i] == 6 or origin_data['issue-function'][i] == 21
            or origin_data['issue-function'][i] == 42 or origin_data['issue-function'][i] == 38 or origin_data['issue-function'][i] == 39
            or origin_data['issue-function'][i] == 40 or origin_data['issue-function'][i] == 11 or origin_data['issue-function'][i] == 9
            or origin_data['issue-function'][i] == 44 or origin_data['issue-function'][i] == 45 or origin_data['issue-function'][i] == 47
            or origin_data['issue-function'][i] == 48 or origin_data['issue-function'][i] == 49 or origin_data['issue-function'][i] == 51
            or origin_data['issue-function'][i] == 53 or origin_data['issue-function'][i] == 54 or origin_data['issue-function'][i] == 55
            or origin_data['issue-function'][i] == 57 or origin_data['issue-function'][i] == 17 or origin_data['issue-function'][i] == 59
            or origin_data['issue-function'][i] == 60 or origin_data['issue-function'][i] == 61 or origin_data['issue-function'][i] == 62
            or origin_data['issue-function'][i] == 64 or origin_data['issue-function'][i] == 65 or origin_data['issue-function'][i] == 67):
                issue_4_data.append(str(origin_data['review'][i]))

        #issue_5 앱 외부
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 22 or origin_data['issue-function'][i] == 5 or origin_data['issue-function'][i] == 41
            or origin_data['issue-function'][i] == 68):
                issue_5_data.append(str(origin_data['review'][i]))

        return issue_0_data, issue_1_data, issue_2_data, issue_3_data, issue_4_data, issue_5_data

    def makeDataFrame(self, issue):
        data = pd.DataFrame.from_records(issue)
        return data

    def counter(self, issue, top5):
        issue_ = []

        for n in range(len(top5)):
            exist = False
            for i in range(len(issue)):
                if issue[0][i] == top5[n]:
                    issue_.append(issue[1][i])
                    exist = True
            if exist == False:
                issue_.append(0)
        
        return issue_

