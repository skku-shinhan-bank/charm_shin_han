class KoBARTConfig:
    def __init__(
        batch_size,
        max_seq_len,
        lr,         #learning_rate
        warmup_ratio,
        num_worker,
        tokenizer_path,
        model_path,
        default_root_dir,
    ):
        batch_size = batch_size
        max_seq_len = max_seq_len
        lr = lr
        warmup_ratio = warmup_ratio
        num_worker = num_worker     #5
        tokenizer_path = tokenizer_path
        model_path = model_path
        default_root_dir = default_root_dir

    
