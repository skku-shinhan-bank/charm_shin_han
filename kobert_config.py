class KoBERTConfig:
  def __init__(
	self,
	num_of_classes,
	max_len,
	batch_size,
	warmup_ratio,
	num_epochs,
	max_grad_norm,
	log_interval,
	learning_rate
	):	
			self.num_of_classes = num_of_classes
			self.max_len = max_len
			self.batch_size = batch_size
			self.warmup_ratio = warmup_ratio
			self.num_epochs = num_epochs
			self.max_grad_norm = max_grad_norm
			self.log_interval = log_interval
			self.learning_rate =  learning_rate
