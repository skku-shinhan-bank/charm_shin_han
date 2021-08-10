class KoBARTConfig:
    def __init__(
        self,
        batch_size,
        max_seq_len,
        lr,         #learning_rate
        warmup_ratio,
        num_workers,
        tokenizer_path,
        model_path,
        default_root_dir,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.num_workers = num_workers     
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.default_root_dir = default_root_dir

    
