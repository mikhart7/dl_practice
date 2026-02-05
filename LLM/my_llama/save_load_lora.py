import torch
from torch import nn
from lora import LoRA_block
import llama2
from lora import add_lora_to_model
def save_lora(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LoRA_block):
            torch.save(module, f"{name}.pth")
            print(f'adapter {name} saved')

def load_lora(model: nn.Module, path_prefix=""):
    for name, module in model.named_modules():
        if isinstance(module, LoRA_block):
            file_path = f"{path_prefix}{name}.pth"
            try:
                loaded_module = torch.load(file_path, map_location="cpu")
            except FileNotFoundError:
                print(f"LoRA adapter {name} not found at {file_path}")
                continue

            module.lin1.weight.data.copy_(loaded_module.lin1.weight.data)
            module.lin2.weight.data.copy_(loaded_module.lin2.weight.data)

            print(f"LoRA adapter {name} loaded from {file_path}")



if __name__ == '__main__':
    batch_size = 16
    n_layers, n_heads, vocab_size = 4, 8, 30522
    hidden_size = 32
       
    
    seq_len = 32
    
    llama = llama2.Llama(llama2.LlamaConfig(n_layers, n_heads, vocab_size, hidden_size))
    
    add_lora_to_model(llama)

    load_lora(llama)