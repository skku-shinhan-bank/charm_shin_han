#-*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# sys.setdefaultencoding('utf-8')

class IssueCounter:
    def __init__(self):
        return
    
    def makecount(self, data_path):   #data_path = /content/drive/MyDrive/신한은행/training-data/Labeled_Data_2/shinhan_app_review_4.xlsx
        origin_data = pd.read_excel(data_path)
        
        issue_0_day_data = []
        issue_1_day_data = []
        issue_2_day_data = []
        issue_3_day_data = []
        issue_5_day_data = []

        #issue_0 실행기능
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 2 or origin_data['issue-function'][i] == 43):
                issue_0_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_0, week_data_0 = self.num_counter(issue_0_day_data, '실행기능')

        #issue_1 로그인
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 0 or origin_data['issue-function'][i] == 1 or origin_data['issue-function'][i] == 12
            or origin_data['issue-function'][i] == 52or origin_data['issue-function'][i] == 16):
                issue_1_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_1, week_data_1 = self.num_counter(issue_1_day_data, '로그인')

        #issue_2 회원가입
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 3 or origin_data['issue-function'][i] == 7):
                issue_2_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_2, week_data_2 = self.num_counter(issue_2_day_data, '회원가입')

        #issue_3 긍융
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 4 or origin_data['issue-function'][i] == 10 or origin_data['issue-function'][i] == 33
            or origin_data['issue-function'][i] == 33 or origin_data['issue-function'][i] == 34 or origin_data['issue-function'][i] == 35
            or origin_data['issue-function'][i] == 36 or origin_data['issue-function'][i] == 37 or origin_data['issue-function'][i] == 15
            or origin_data['issue-function'][i] == 50 or origin_data['issue-function'][i] == 46 or origin_data['issue-function'][i] == 56
            or origin_data['issue-function'][i] == 58 or origin_data['issue-function'][i] == 18 or origin_data['issue-function'][i] == 63
            or origin_data['issue-function'][i] == 66):
                issue_3_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_3, week_data_3 = self.num_counter(issue_3_day_data, '금융')

        #issue_5 앱 외부
        for i in range (len(origin_data)):
            if(origin_data['issue-function'][i] == 22 or origin_data['issue-function'][i] == 5 or origin_data['issue-function'][i] == 41
            or origin_data['issue-function'][i] == 68):
                issue_5_day_data.append(str(origin_data['일자'][i])[:10])

        month_data_5, week_data_5 = self.num_counter(issue_5_day_data, '앱외부')


        #count표 만들기
        #월별
        data_month = pd.concat([month_data_0, month_data_1, month_data_2, month_data_3, month_data_5], axis=1)
        data_month = data_month.fillna(0)

        #주별
        data_week = pd.concat([week_data_0, week_data_1, week_data_2, week_data_3, week_data_5], axis=1)
        data_week = data_week.fillna(0)

        #plot
        print("월별 이슈 분포\n")
        self.show(data_week)

        print("\n주별 이슈 분포\n")
        self.show(data_week)

        #json
        month_json0 = self.tojson(data_month, '실행기능')
        month_json1 = self.tojson(data_month, '로그인')
        month_json2 = self.tojson(data_month, '회원가입')
        month_json3 = self.tojson(data_month, '금융')
        month_json4 = self.tojson(data_month, '앱외부')
        week_json0 = self.tojson(data_week, '실행기능')
        week_json1 = self.tojson(data_week, '로그인')
        week_json2 = self.tojson(data_week, '회원가입')
        week_json3 = self.tojson(data_week, '금융')
        week_json4 = self.tojson(data_week, '앱외부')

        return data_month, data_week, month_json0, month_json1, month_json2, month_json3, month_json4, week_json0, week_json1, week_json2, week_json3, week_json4

    def num_counter(self, data, issue):
        data = pd.DataFrame(data)
        data = data.rename(columns={0:'datetime'})
        data['date'] = data['datetime'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
        data.set_index(data['date'], inplace=True)
        data = data.drop('datetime',1)
        data = data.drop('date',1)
        data[issue] = 1
        month = data.resample('M').sum()
        month = month.fillna(0)
        week = data.resample('W').sum()
        week = week.fillna(0)

        return month, week
        
    def tojson(self, data, issue):
        date = [str(item)[:10] for item in data.index]
        date = pd.DataFrame(date)
        index = [i for i in range(len(date))]
        issue0 = data[[issue]]
        issue0 = pd.DataFrame(issue0)
        issue0.set_index(keys=[index], inplace=True)
        issue0_data = pd.concat([issue0, date], axis=1)
        issue0_data = issue0_data.rename(columns={0:"x_label"})
        issue0_data = issue0_data.rename(columns={issue:"y"})
        issue0_data = issue0_data.rename_axis('x')
        
        js = issue0_data.to_json(orient = 'table')
        return js
    
    def show(self, data):       
        data.plot()