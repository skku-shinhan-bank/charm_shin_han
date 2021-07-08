class KoElectraConfig:
  def __init__(
	self,
	num_of_train_data,
	num_of_classes,
	max_len,
	batch_size,
	num_epochs,
	learning_rate
	):	
			self.num_of_classes = num_of_classes
			self.num_of_train_data = num_of_train_data
			self.max_len = max_len
			self.batch_size = batch_size
			self.num_epochs = num_epochs
			self.learning_rate =  learning_rate