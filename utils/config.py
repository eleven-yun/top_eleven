# Change the config file: .py->.json
top_config = {
    "opt_params": {
        "lr": 0.01,
        "weight_decay": 1e-5,
        "eps": 1e-4
    },
    "train_params": {
        "batch_size": 5,
        "epoch": 30,
        "warmup": 100,
        "save_path": "./saved"
    },
    "model_params": {
        "num_layers": 6,
        "d_model": 512,
        "nhead": 8,
        "enc_length": 10,
        "src_length": 50,
        "num_classes": 3,
    }
}
