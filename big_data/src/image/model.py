import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.nn.init import trunc_normal_
import open_clip


def get_encoder(args):

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
    model_dim = 768 # BioCLIP-2 uses ViT-L/14 → 768 dim

    model = model.to(args.device)

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["attn"],
        lora_dropout=args.lora_dropout,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model, preprocess_train, preprocess_val, tokenizer, model_dim  


class HashCoder(nn.Module):
    """
    Flexible MLP hashing network with 'small' and 'large' variants.
    """
    def __init__(self, in_dim, bitdim, hidden_dim=2048, hashcoder="small"):
        super().__init__()
        self.hashcoder = hashcoder.lower()
        self.bitdim = bitdim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        if self.hashcoder == "small":
            layers = [
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, bitdim),
                nn.BatchNorm1d(bitdim)
            ]
        elif self.hashcoder == "large":
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, bitdim),
                nn.BatchNorm1d(bitdim)
            ]
        else:
            raise ValueError(f"Unknown HashCoder variant: {hashcoder}. Choose ['small','large'].")

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)
    

def get_hashcoder(in_dim, args):
    """
    Returns a HashCoder MLP with proper device placement.
    """
    hashcoder = HashCoder(in_dim=in_dim, bitdim=args.bitdim, hidden_dim=args.hidden_dim, hashcoder=args.hashcoder)
    return hashcoder.to(args.device)