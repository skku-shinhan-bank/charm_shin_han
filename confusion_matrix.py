class ConfusionMatrix:
  def __init__(self, column):
    self.matrix = []
    for i in range(column):
      self.matrix.append([])
      for j in range(column):
       self. matrix[i].append(0)
  def add(self, real_class_id, predict_class_id):
    self.matrix[real_class_id][predict_class_id] += 1
  def show(self):
    for row in self.matrix:
      print(row)