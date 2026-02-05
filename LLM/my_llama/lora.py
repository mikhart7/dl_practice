import torch
from torch import nn

class LoRA_block(nn.Module):
    def __init__(self, input_dim, output_dim, r=8):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, r, bias=False)
        self.lin2 = nn.Linear(r, output_dim, bias=False)

        torch.nn.init.xavier_normal_(self.lin1.weight)
        torch.nn.init.zeros_(self.lin2.weight)

    def forward(self, x):
        return self.lin2(self.lin1(x))

class LoRAWrapper(nn.Module):
    def __init__(self, module: nn.Linear, r=8):
        super().__init__()
        self.module = module
        self.module.requires_grad_(False)
        self.lora = LoRA_block(module.in_features, module.out_features, r=r)

    def forward(self, x):
        return self.module(x) + self.lora(x)
    

def add_lora_to_model(model: nn.Module, target_modules=("WQ", "WK"), r=8):
    for name, module in model.named_modules():
      if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
          parent = model
          name_parts = name.split('.')
          for part in name_parts[:-1]:
              parent = getattr(parent, part)
          setattr(parent, name_parts[-1], LoRAWrapper(module, r=8))

