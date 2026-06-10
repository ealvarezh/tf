import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Deep Hashing Training")

    # -----------------------
    # General
    # -----------------------
    parser.add_argument("--wandb", action="store_true", help="Save runs in wandb")
    parser.add_argument("--entity", type=str, default='', help="Entity name for wandb")
    parser.add_argument("--projname", type=str, default='', help="Project name for wandb")
    parser.add_argument("--tmpdir", type=str, default='./tmp', help="Dir for logging tmp runs")
    parser.add_argument("--ckpt_dir", type=str, default='./ckpt', help="Directory to save model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on")

    # -----------------------
    # Data
    # -----------------------
    parser.add_argument("--data_dir", type=str, default="", help="Directory of the dataset")
    parser.add_argument("--n_workers", type=int, default=32, help="Number of dataloader workers")
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size for training/evaluation")
    parser.add_argument("--resize_size", type=int, default=256, help="Resize size before center crop for evaluation")
    parser.add_argument("--scale_min", type=float, default=0.4, help="Minimum scale for random resized crop")

    # -----------------------
    # Model
    # -----------------------
    parser.add_argument("--hashcoder", type=str, default="large", choices=["small", "large"], help="Hashcoder variant")
    parser.add_argument("--nlayers", type=int, default=3, help="Number of layers in HashCoder")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension in HashCoder")
    parser.add_argument("--bitdim", type=int, default=256, help="Number of bits for HashCoder")


    # -----------------------
    # Training
    # -----------------------
    parser.add_argument("--train_mini", action="store_true", help="use train mini split")
    parser.add_argument("--frozenepochs", type=int, default=1, help="Epochs for head pretraining before LoRA")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (for r and alpha)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--coeff_inv", type=float, default=1.0, help="Weight for invariance (augmentation) loss")
    parser.add_argument("--coeff_reg", type=float, default=1.0, help="Weight for regularization loss")


    # -----------------------
    # Evaluation
    # -----------------------
    parser.add_argument("--evaldataset", type=str, default="", help="Evaluation dataset")
    parser.add_argument("--ckpt", type=str, default="", help="Path to the model checkpoint")


    # -----------------------
    # Optimization
    # -----------------------
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Linear LR warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate after warmup")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate at the end of schedule")
    parser.add_argument("--wd", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--eps", type=float, default=0.05, help="Epsilon for total coding rate regularization")

    return parser.parse_args()


args = get_args()