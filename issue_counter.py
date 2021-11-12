#-*- coding:utf-8 -*-
import sys
import collections
import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
plt.style.use('ggplot')
sys.setdefaultencoding('utf-8')

class IssueCounter:
    def __init__(self):
        return
    
    def makecount(self, data_path):   #data_path = /content/drive/MyDrive/신한은행/training-data/Labeled_Data_2/shinhan_app_review_4.xlsx
        origin_data = pd.read_excel(data_path)
        
        issue_0_month_data = []
        issue_0_day_data = []
        issue_1_month_data = []
        issue_1_day_data = []
        issue_2_month_data = []
        issue_2_day_data = []
        issue_3_month_data = []
        issue_3_day_data = []
        issue_4_month_data = []
        issue_4_day_data = []
        issue_5_month_data = []
        issue_5_day_data = []

        #issue_0 실행기능
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 2 or origin_data['issue-function'][i] == 43):
                issue_0_month_data.append(str(origin_data['일자'][i])[:7])
                issue_0_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_0 = self.num_counter(issue_0_month_data, '실행기능')
        day_data_0 = self.num_counter(issue_0_day_data, '실행기능')

        #issue_1 로그인
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 0 or origin_data['issue-function'][i] == 1 or origin_data['issue-function'][i] == 12
            or origin_data['issue-function'][i] == 52or origin_data['issue-function'][i] == 16):
                issue_1_month_data.append(str(origin_data['일자'][i])[:7])
                issue_1_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_1 = self.num_counter(issue_1_month_data, '로그인')
        day_data_1 = self.num_counter(issue_1_day_data, '로그인')

        #issue_2 회원가입
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 3 or origin_data['issue-function'][i] == 7):
                issue_2_month_data.append(str(origin_data['일자'][i])[:7])
                issue_2_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_2 = self.num_counter(issue_2_month_data, '회원가입')
        day_data_2 = self.num_counter(issue_2_day_data, '회원가입')

        #issue_3 긍융
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 4 or origin_data['issue-function'][i] == 10 or origin_data['issue-function'][i] == 33
            or origin_data['issue-function'][i] == 33 or origin_data['issue-function'][i] == 34 or origin_data['issue-function'][i] == 35
            or origin_data['issue-function'][i] == 36 or origin_data['issue-function'][i] == 37 or origin_data['issue-function'][i] == 15
            or origin_data['issue-function'][i] == 50 or origin_data['issue-function'][i] == 46 or origin_data['issue-function'][i] == 56
            or origin_data['issue-function'][i] == 58 or origin_data['issue-function'][i] == 18 or origin_data['issue-function'][i] == 63
            or origin_data['issue-function'][i] == 66):
                issue_3_month_data.append(str(origin_data['일자'][i])[:7])
                issue_3_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_3 = self.num_counter(issue_3_month_data, '금융')
        day_data_3 = self.num_counter(issue_3_day_data, '금융')

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
                issue_4_month_data.append(str(origin_data['일자'][i])[:7])
                issue_4_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_4 = self.num_counter(issue_4_month_data, '기타')
        day_data_4 = self.num_counter(issue_4_day_data, '기타')

        #issue_5 앱 외부
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 22 or origin_data['issue-function'][i] == 5 or origin_data['issue-function'][i] == 41
            or origin_data['issue-function'][i] == 68):
                issue_5_month_data.append(str(origin_data['일자'][i])[:7])
                issue_5_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_5 = self.num_counter(issue_5_month_data, '앱외부')
        day_data_5 = self.num_counter(issue_5_day_data, '앱외부')

        #count표 만들기
        data_month = pd.concat([month_data_0, month_data_1, month_data_2, month_data_3, month_data_4, month_data_5], axis=1)
        month_data = pd.DataFrame(data_month, columns=['실행기능', '로그인', '회원가입', '금융', '기타', '앱외부', 'x_label_금융'])
        month_data = month_data.rename(columns={'x_label_금융':'x_label'})
        month_data.index.name='x'
        month_data = month_data.fillna(0)

        data_day = pd.concat([day_data_0, day_data_1, day_data_2, day_data_3, day_data_4, day_data_5], axis=1)
        day_data = pd.DataFrame(data_day, columns=['실행기능', '로그인', '회원가입', '금융', '기타', '앱외부', 'x_label_로그인'])
        day_data = day_data.rename(columns={'x_label_로그인':'x_label'})
        day_data.index.name='x'
        day_data = day_data.fillna(0)

        month_json = self.tojson(month_data)
        day_json = self.tojson(day_data)

        print("월별 이슈 분포\n")
        x_label = pd.Series([str(i) for i in month_data['x_label']])
        Month = pd.DataFrame(month_data, columns=['실행기능', '로그인', '회원가입', '금융', '기타', '앱외부'])
        Month = Month.set_index(keys=[x_label], inplace=False)
        self.show(Month)

        print("\n일별 이슈 분포\n")
        xx_label = pd.Series([str(i) for i in day_data['x_label']])
        Day = pd.DataFrame(day_data, columns=['실행기능', '로그인', '회원가입', '금융', '기타', '앱외부'])
        Day = Day.set_index(keys=[xx_label], inplace=False)
        self.show(Day)

        return month_data, day_data, month_json, day_json


    def num_counter(self, data, issue):
        data = collections.Counter(data)
        sorted_data = sorted(data.items(), key=itemgetter(1), reverse=True)
        sorted_data = sorted(sorted_data)

        filePath = './'+issue+'.csv'

        with open(filePath, 'w', encoding='UTF-8') as lf:
            lf.write("x_label,y\n")
            for key, value in data.items():
                lf.write('{},{}\n'.format(key, value))
        lf.close()

        file = pd.read_csv(filePath)
        file.to_excel('month_'+issue+'.xlsx')
        Data = pd.read_excel('./month_'+issue+'.xlsx')
        Month = pd.DataFrame(Data, columns=['y', 'x_label'])
        Month = Month.rename(columns={'y':issue})
        Month = Month.rename(columns={'x_label':'x_label_'+issue})

        return Month
        
    def tojson(self, data):
        js = data.to_json(orient = 'table')
        return js
    
    def show(self, data):       
        data.plot()