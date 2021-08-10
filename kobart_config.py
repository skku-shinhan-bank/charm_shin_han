class KoBARTConfig:
    def __init__(
        batch_size,
        max_seq_len,
        lr,
        warmpup_ratio,
        num_worker,
        tokenizer_path,
        model_path,
        default_root_dir,
        ):

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.lr = lr
        self.warmup_ratio = warmpup_ratio
        self.num_worker = num_worker     #5
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.default_root_dir = default_root_dir

    
