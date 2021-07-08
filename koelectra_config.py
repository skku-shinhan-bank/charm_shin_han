class KoELECTRAConfig:
  def __init__(
    self,
    num_of_train_data,
    n_epoch,       # Num of Epoch
    batch_size,      # 배치 사이즈
    save_step,   # 학습 저장 주기
    num_label,
    max_seq_len,
	):	
    self.num_of_train_data = num_of_train_data
    self.n_epoch = n_epoch
    self.batch_size = batch_size
    self.save_step = save_step
    self.num_label = num_label
    self.max_seq_len = max_seq_len
