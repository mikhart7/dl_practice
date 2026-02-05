import torch 
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", clean_up_tokenization_spaces=True)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def collate_fn(
    tokenizer: AutoTokenizer, batch: list[str]
) -> tuple[torch.Tensor,  torch.Tensor]:
    encoded_batch = tokenizer.batch_encode_plus(
        batch, padding="longest", return_tensors="pt", return_token_type_ids=False)
    return encoded_batch.to(device)


def get_loaders(path = "blo05/cleaned_wiki_en_20-40", from_csv=False, batch_size=8):
    if from_csv:
        ds = load_dataset("csv", data_files=path)['train'].filter(lambda x: len(x['text']) <=1500 )
    else:
        ds = load_dataset(path)['train'].filter(lambda x: len(x['text']) <=1500 )

    ds_split = ds.train_test_split(test_size=0.2, seed=777)

    ds_train =  ds_split['train']['text']
    ds_val =  ds_split['test']['text']

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=lambda batch:collate_fn(tokenizer,batch))
    val_loader = DataLoader(ds_val, batch_size=batch_size, collate_fn=lambda batch:collate_fn(tokenizer,batch))
    
    return train_loader, val_loader

    

if __name__ == '__main__':
    train_loader, val_loader = get_loaders("cleaned_wiki_en_small.csv", from_csv=True)

    print(next(iter(train_loader)))
