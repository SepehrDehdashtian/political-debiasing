"""Hate-Demographic split of Yoder's Hate Speech Dataset."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
import datasets
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
import torch
from tqdm.auto import tqdm 

__all__ = ['HateDemLoader']

class PrepareData:
    def __init__(self, opts):
        self.opts = opts
        
        if opts.dataset_options["language_model"] == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").cuda()
            self.feature_file_extension = "distilbert_features"
            
        elif opts.dataset_options["language_model"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.encoder = BertModel.from_pretrained("bert-base-uncased").cuda()
            self.feature_file_extension = "bert_features"
            
        elif opts.dataset_options["language_model"] == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.encoder = RobertaModel.from_pretrained("roberta-base").cuda()
            self.feature_file_extension = "roberta_features"
            
        else:
            raise ValueError(opts.dataset_options["language_model"])
        
        # self.encode_mode = "pool"
        self.encode_mode = "cls"

        if self.encode_mode == "pool":
            self.feature_file_extension += "_pool.pt"
        else:
            self.feature_file_extension += ".pt"
    
    
    def _tokenize(self, text):
        # return self.tokenizer(text, add_special_tokens=True, return_tensors="pt", padding=True)
        return self.tokenizer(text, add_special_tokens=True, padding="max_length", max_length=64, return_attention_mask=True, truncation="longest_first", return_tensors="pt")
    
    def _encode(self, input_ids, attention_mask):
        if self.encode_mode == "pool":
            return self.encoder(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())[1]
        else:
            return self.encoder(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda()).last_hidden_state[0, 0]
        
    @torch.no_grad()
    def process_split(self, df, batch_size=256):
        x_out = torch.zeros((df.shape[0], 768), device='cuda:0')
        N = df.shape[0] // batch_size
        
        for i in tqdm(range(N)):
            text = df.iloc[i * batch_size: (i + 1) * batch_size]["text"]
            x = self._tokenize(text.values.tolist())
            # import pdb; pdb.set_trace()
            x_out[i * batch_size: (i + 1) * batch_size, :] = self._encode(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
            
        y = torch.from_numpy(df["y"].values).reshape(-1)
        s = torch.from_numpy(df["s"].values).reshape(-1)
        return {"x": x_out.cpu(), "y": y, "s": s}
            
    def load_data(self) -> dict:
        feature_file = f"{os.path.dirname(self.opts.dataset_options['path'])}/{self.feature_file_extension}"

        if os.path.exists(feature_file):
            print('Loading the features...')
            data = torch.load(feature_file, map_location='cpu')
        else:
            data_df = datasets.load_dataset("json", data_files=self.opts.dataset_options["path"])["train"].to_pandas()
            
            data_df["s"] = np.random.randint(0, 2, (data_df.shape[0],))
            data_df["y"] = data_df["hate"].astype(int)
            
            cols = ["text", "y", "s"]
            data = dict()
            data["train"] = data_df.loc[data_df["fold"] == "train", cols]
            data["test"] = data_df.loc[data_df["fold"] == "test", cols]
            
            data["train"], data["val"] = train_test_split(data["train"], test_size=0.1, shuffle=True, stratify=data["train"]["y"], random_state=0)
            
            for split in data.keys():
                data[split] = data[split].reset_index()
                
                print(f"Tokenizing and extracting features from {split}")
                data[split] = self.process_split(data[split])

            torch.save(data, f"{os.path.dirname(self.opts.dataset_options['path'])}/{self.feature_file_extension}")
            print(f"Saved encoded features.")
            
        return data

class HateDemDataloader:
    def __init__(self, data, opts):
        self.opts = opts
        self.data = data
        
    def __len__(self):
        return len(self.data['y'])

    @torch.no_grad()
    def __getitem__(self, index):
        
        # try:
        x = self.data["x"][index]
        y = self.data["y"][index]
        s = self.data["s"][index]
        # except Exception as e:
        #     print(e)
        #     print(index, len(self.data["x"]), len(self.data["y"]), len(self.data["s"]))

        return x, y, s

 
class HateDemLoader(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

        pre = PrepareData(opts)
        self.data = pre.load_data()


    def train_dataloader(self):
        dataset = HateDemDataloader(self.data['train'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self):
        dataset = HateDemDataloader(self.data['val'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self):
        dataset = HateDemDataloader(self.data['test'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader


    def train_kernel_dataloader(self):
        idx_sampled = torch.randperm(len(self.data['train']['y']))[:self.opts.dataset_options["kernel_numSamples"]]
        x      = self.data['train']['x'][idx_sampled].cuda()
        y      = self.data['train']['y'][idx_sampled].cuda()
        s      = self.data['train']['s'][idx_sampled].cuda()
        return x, y, s