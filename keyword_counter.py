import collections
import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class KeywordCouter:
    def __init__(self):
        return
    
    def makecount(self, issue0, issue1, issue2, issue3, issue4, issue5):  #list
        top = issue0['0'][:5]+issue1

        return

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
                issue_2_data.append(str(origin_data['reveiw'][i]))

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

    

