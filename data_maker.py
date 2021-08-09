import pandas as pd

class DataMaker:
	@staticmethod
	def make_shinhan_issue_class_data(file_path, test_index_file_path):
		data_frame = pd.read_excel(file_path)

		f = open(test_index_file_path, 'r')
		data = f.read()

		test_index_list = []
		test_index_list = data.split(', ')
		f.close()
		test_index_list = list(map(int, test_index_list))

		data = []
		label = []

		for index, row in data_frame.iterrows():
			test_issue_id = row['issue-function']
			test_review = row['review']

			if test_issue_id == 2 or test_issue_id == 43:
				label.append(0)
			elif test_issue_id == 0 or test_issue_id == 1 or test_issue_id == 12 or test_issue_id == 52 or test_issue_id == 16:
				label.append(1)
			elif test_issue_id == 3 or test_issue_id == 7:
				label.append(2)
			elif test_issue_id == 4 or test_issue_id == 10 or test_issue_id == 33 or test_issue_id == 34 or test_issue_id == 35 or test_issue_id == 36 or test_issue_id == 37  or test_issue_id == 15 or test_issue_id == 50 or test_issue_id == 46 or test_issue_id == 56 or test_issue_id == 58 or test_issue_id == 18 or test_issue_id == 63 or test_issue_id == 66 :
				label.append(3)
			elif test_issue_id == 13 or test_issue_id == 14 or test_issue_id == 26 or test_issue_id == 8 or test_issue_id == 6 or test_issue_id == 21 or test_issue_id == 42 or test_issue_id == 38  or test_issue_id == 39 or test_issue_id == 40 or test_issue_id == 11 or test_issue_id == 9 or test_issue_id == 44 or test_issue_id == 45 or test_issue_id == 47 or test_issue_id == 48 or test_issue_id == 49 or test_issue_id == 51 or test_issue_id == 53 or test_issue_id == 54 or test_issue_id == 55 or test_issue_id == 57 or test_issue_id == 17 or test_issue_id == 59 or test_issue_id == 60 or test_issue_id == 61 or test_issue_id == 62 or test_issue_id == 64 or test_issue_id == 65 or test_issue_id == 67 :
				label.append(4)
			elif test_issue_id == 22 or test_issue_id == 5 or test_issue_id == 41 or test_issue_id == 68:
				label.append(5)
			elif test_issue_id == 23:
				label.append(6)
			else:
				break
			data.append(test_review)

		train_data = []
		train_label = []
		test_data = []
		test_label = []

		for i in range(len(data)):
			if i in test_index_list:
				test_data.append(data[i])
				test_label.append(label[i])
			else:
				train_data.append(data[i])
				train_label.append(label[i])

		return train_data, train_label, test_data, test_label
	
	@staticmethod
	def make_issue_class_data_from_crawled(file_path):
		data_frame = pd.read_excel(file_path)
		
		data = []
		label = []

		for index, row in data_frame.iterrows():
			test_issueId = row[1]
			test_review = row[0]
			label.append(test_issueId)
			data.append(test_review)
		
		return data, label