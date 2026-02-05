import sys
import math
import torch 
from dataclasses import dataclass

from torch import nn
from torch.nn import Linear

import torch.nn.functional as F

from lora import add_lora_to_model
from save_load_lora import load_lora
@dataclass
class LlamaConfig():
    n_layers: int
    n_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int = 2000  # Нужно для RoPe, чтобы не создавть RoPe при новом проходе заново, а брать углы поворота до seq_len.
    

class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        self.n_layers = config.n_layers

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.llama_layers = nn.ModuleList([LlamaLayer(config) for i in range(self.n_layers)])
        self.rms_norm = nn.RMSNorm(config.hidden_size)
        self.logits_layer = LogitsLayer(config)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.llama_layers:
            x = layer(x)

        return self.logits_layer(self.rms_norm(x))


class LogitsLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.W = Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):  
        return self.W(x)


class LlamaLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.self_attention = SelfAttention(config)
        self.ffn = FFN(config)
        self.rms_norm = nn.RMSNorm(config.hidden_size)

    def forward(self, x):
        # x.shape = (B, seq_len, hidden_dim
        x_normed = self.rms_norm(x)
        x = x + self.self_attention(x_normed)

        x_normed = self.rms_norm(x)
        x = x + self.ffn(x)

        return x
    

class FFN(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.W1 = Linear(config.hidden_size, config.hidden_size)
        self.W2 = Linear(config.hidden_size, config.hidden_size)
        self.W3 = Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, x):
        return self.W3(F.silu(self.W1(x)) * self.W2(x))
    

class SelfAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.WQ = Linear(config.hidden_size, config.hidden_size)
        self.WK = Linear(config.hidden_size, config.hidden_size)
        self.WV = Linear(config.hidden_size, config.hidden_size)
        self.WO = Linear(config.hidden_size, config.hidden_size)
        self.RoPe = RoPE(config)

    def forward(self, x):
        B, seq_len, hidden_size = x.shape
        
        q = self.WQ(x)
        k = self.WK(x)
        v = self.WV(x)
        q = q.view(B, seq_len, self.n_heads, self.head_dim)
        k = k.view(B, seq_len, self.n_heads, self.head_dim)
        v = v.view(B, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.RoPe(q, k)
        
        attn = nn.functional.softmax(
            torch.where(
                torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(torch.bool), #Главную диагональ не маскируем
                -torch.inf,
                (q @ k.transpose(-2, -1)) * (1 / math.sqrt(self.head_dim)),
            ),
            dim=-1,
        )
        
        out = self.WO((attn @ v).transpose(1,2).reshape(B, seq_len, hidden_size))
        return out

class RoPE(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.base = 10000.0
        self.head_dim = config.hidden_size // config.n_heads
        self.t = self.base ** (- 2 * (torch.arange(0, self.head_dim // 2, dtype=torch.float32) - 1) / (self.head_dim))
        self.range = torch.arange(0, self.max_seq_len, dtype=torch.float32)

        self.temp = self.range.view(-1, 1) * self.t.view(1, -1)

        self.cos = torch.cos(self.temp)
        self.sin = torch.sin(self.temp)

    def split(self, x):
        # x.shape = (B, n_head, seq_len, head_dim)

        first_half = x[..., :self.head_dim // 2]
        second_half = x[..., self.head_dim // 2:]
        return first_half, second_half
    
    def rotate(self, x):
        # x.shape = (B, n_head, seq_len, head_dim)

        first_half, second_half = self.split(x)
        seq_len = x.size(-2)
        return torch.cat(
            [
                first_half * self.sin[:seq_len] + second_half * self.cos[:seq_len],
                first_half * self.cos[:seq_len] - second_half * self.sin[:seq_len]
            ], 
            dim=-1
        )

    def forward(self, q, k):
        # (q|k).shape = (B, n_head, seq_len, head_dim)

        return self.rotate(q), self.rotate(k)

class RMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.scale = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x):
        return (x * torch.rsqrt((x**2).mean(dim = -1, keepdim=True))) * self.scale
        
 
 

if __name__ == '__main__':
    if len(sys.argv) == 1:
        batch_size = 16
        n_layers, n_heads, vocab_size = 4, 8, 30522
        hidden_size = 32
       
    else:
        batch_size, n_layers, n_heads, vocab_size, hidden_size = map(int, sys.argv[1:]) 
    
    seq_len = 32
    
    llama = Llama(LlamaConfig(n_layers, n_heads, vocab_size, hidden_size))
    
    add_lora_to_model(llama)

    load_lora(llama)



    optimizer = torch.optim.Adam(llama.parameters(), lr=0.005)


    ce_loss = nn.CrossEntropyLoss()
        
    input = torch.randint(0, vocab_size, size=(batch_size, seq_len))

    logits = llama(input)

    logits = llama(input)[:,:-1,:]
    targets = input[:,1:]

    loss = ce_loss(logits.reshape(-1, vocab_size), targets.reshape(-1))

    # #loss = logits.mean()
    loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    # print(llama.modules)
    print('logits shape = ', list(logits.shape))

