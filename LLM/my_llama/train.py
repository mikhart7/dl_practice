import torch 
from torch import nn
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from llama2 import Llama, LlamaConfig
import data
from lora import add_lora_to_model
from save_load_lora import save_lora

def train_epoch(model: Llama, loader, optimizer, loss_fun, writer, print_every, n_epoch):
    epoch_loss = 0 
    global_step = 0
    length = len(loader)
    for batch in tqdm(loader):
        input = batch['input_ids']

        logits = model(input)[:,:-1,:].contiguous()
        targets = input[:,1:].contiguous()
        vocab_size = logits.size(-1)
        loss = loss_fun(logits.reshape(-1, vocab_size), targets.reshape(-1))

        epoch_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        
        if global_step % print_every == 0:
            print(f"[step {global_step}] train_loss={loss.item():.4f}")
            if writer:
                writer.add_scalar("loss/train", loss.item(), global_step + n_epoch*length)

        global_step += 1
    
    return epoch_loss // len(loader)

@torch.no_grad()
def val_epoch(model: Llama, loader, loss_fun):
    model.eval()
    epoch_loss = 0 
    for batch in tqdm(loader):
        input = batch['input_ids']

        logits = model(input)[:,:-1,:].contiguous()
        targets = input[:,1:].contiguous()
        vocab_size = logits.size(-1)

        loss = loss_fun(logits.reshape(-1, vocab_size), targets.reshape(-1))

        epoch_loss += loss
        
    return epoch_loss // len(loader)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def train(cfg: DictConfig):
    writer = log(cfg)
    llama = Llama(LlamaConfig(cfg.model.n_layers, cfg.model.n_heads, cfg.model.vocab_size, cfg.model.hidden_size))
    if cfg.model.with_lora:
        add_lora_to_model(llama)

    train_loader, val_loader = data.get_loaders("cleaned_wiki_en_small.csv", from_csv=True, batch_size=cfg.train.batch_size) #"cleaned_wiki_en_small.csv", from_csv=True

    optimizer = torch.optim.Adam(llama.parameters(), cfg.train.lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=data.tokenizer.pad_token_id)

    for epoch in range(cfg.train.epochs):
        epoch_loss = train_epoch(llama, train_loader, optimizer, ce_loss, writer, cfg.logging.print_every, epoch)
        val_loss =  val_epoch(llama, val_loader, ce_loss)
        if writer:
            writer.add_scalar("epoch_loss/train", epoch_loss.item(), epoch)  
            writer.add_scalar("epoch_loss/val", val_loss.item(), epoch) 

    
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}")

    if cfg.model.with_lora:
        save_lora(llama)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def log(cfg: DictConfig):
    writer = None

    if cfg.logging.tensorboard:
        log_dir = os.path.join(cfg.logging.save_dir, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[TensorBoard] logging to {log_dir}")
    return writer

if __name__ == '__main__':
    train()


