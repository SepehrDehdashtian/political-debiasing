from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl



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

__all__ = ['Reddit']

class PrepareRedditData:
    def __init__(self, opts):
        self.opts = opts
        
        if opts.dataset_options["language_model"] == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").cuda()
            self.feature_file_extension = "distilbert_features_pool.pt"
            
        elif opts.dataset_options["language_model"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.encoder = BertModel.from_pretrained("bert-base-uncased").cuda()
            self.feature_file_extension = "bert_features_pool.pt"
            
        elif opts.dataset_options["language_model"] == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.encoder = RobertaModel.from_pretrained("roberta-base").cuda()
            self.feature_file_extension = "roberta_features_pool.pt"
            
        else:
            raise ValueError(opts.dataset_options["language_model"])
        
    def _tokenize(self, text):
        # return self.tokenizer(text, add_special_tokens=True, return_tensors="pt", padding=True)
        return self.tokenizer(text, add_special_tokens=True, padding="max_length", max_length=64, return_attention_mask=True, truncation="longest_first", return_tensors="pt")
    
    def _encode(self, input_ids, attention_mask):
        # return self.encoder(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda()).last_hidden_state[0, 0]
        return self.encoder(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())[1]
    
    @torch.no_grad()
    def process_lines(self, lines, labels, batch_size: int = 256):
        N = len(lines)
        x_out = torch.zeros((N, 768), device="cuda:0")
        B = N // batch_size
        
        for i in tqdm(range(B)):
            text = lines[i * batch_size: (i + 1) * batch_size] # list of str
            x = self._tokenize(text)
            x_out[i * batch_size: (i + 1) * batch_size, :] = self._encode(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
            
        y = torch.from_numpy(labels).long()
        return {"x": x_out.cpu(), "y": y.cpu()}
            
    def load_data(self) -> dict:
        data_dir = self.opts.dataset_options["path"]
        data_left_path = f"{data_dir}/reddit_left_posttrump.txt"
        data_right_path = f"{data_dir}/reddit_right_posttrump.txt"
        
        feature_path = f"{data_dir}/{self.feature_file_extension}"

        if os.path.exists(feature_path):
            print('Loading pre-computed left features...')
            data = torch.load(feature_path, map_location='cpu')
        else:
            data = dict()
            
            with open(data_left_path, "r") as f:
                lines_left = f.readlines()
            with open(data_right_path, "r") as f:
                lines_right = f.readlines()
            N, M = len(lines_left), len(lines_right)
            
            lines = lines_left + lines_right
            labels = np.zeros((N + M,), dtype=np.int32)
            labels[N:] = 1
            
            # lines_train, lines_val, labels_train, labels_val = train_test_split(lines, labels, test_size=0.1, shuffle=True, random_state=0)
            lines_train, lines_val, labels_train, labels_val = train_test_split(lines, labels, test_size=0.1, shuffle=True, random_state=0)
            
            data["train"] = self.process_lines(lines=lines_train, labels=labels_train)
            data["val"] = self.process_lines(lines=lines_val, labels=labels_val)    

            torch.save(data, f"{self.opts.dataset_options['path']}/{self.feature_file_extension}")
            print("Saved computed features")
            
        return data


class RedditDataloader:
    def __init__(self, data, opts):
        self.opts = opts
        self.data = data
        
    def __len__(self):
        return len(self.data["y"])

    @torch.no_grad()
    def __getitem__(self, index):
        
        x = self.data["x"][index]
        y = self.data["y"][index]
        return x, y, 0

 
class Reddit(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

        pre = PrepareRedditData(opts)
        self.data = pre.load_data()


    def train_dataloader(self):
        dataset = RedditDataloader(self.data['train'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self):
        dataset = RedditDataloader(self.data['val'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    # def test_dataloader(self):
    #     dataset = RedditDataloader(self.data['test'], self.opts)

    #     loader = DataLoader(
    #         dataset=dataset,
    #         batch_size=self.opts.batch_size_test,
    #         shuffle=False,
    #         num_workers=self.opts.nthreads,
    #         pin_memory=self.pin_memory
    #     )
    #     return loader


    # def train_kernel_dataloader(self):
    #     idx_sampled = torch.randperm(len(self.data['train']['y']))[:self.opts.dataset_options["kernel_numSamples"]]
    #     x      = self.data['train']['x'][idx_sampled].cuda()
    #     y      = self.data['train']['y'][idx_sampled].cuda()
    #     s      = self.data['train']['s'][idx_sampled].cuda()