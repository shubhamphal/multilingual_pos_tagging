#REFERENCES https://github.com/bentrevett/pytorch-pos-tagging

from collections import Counter

import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchtext.vocab
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import time
import functools
from multilingual_model import BERTPoSTagger
import numpy as np
import random
import os, sys
import argparse
import json

import warnings

from udpos import UDPOS

warnings.filterwarnings("ignore")

# set command line options
parser = argparse.ArgumentParser(description="main.py")
parser.add_argument(
    "--mode",
    type=str,
    choices=["train", "eval"],
    default="train",
    help="Run mode",
)
parser.add_argument(
    "--lang",
    type=str,
    choices=["en", "cs", "es", "ar", "hy", "lt", "af", "ta"],
    default="en",
    help="Language code",
)
parser.add_argument(
    "--model-name",
    type=str,
    default=None,
    help="name of the saved model",
)
args = parser.parse_args()

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

if args.model_name is None:
    args.model_name = "{}-bert-model".format(args.lang)

# set a fixed seed for reproducibility
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
params = json.load(open("config.json"))

# Modify this if you have multiple GPUs on your machine
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
#print(UD_TAGS.vocab.stoi)


def main():
    print("Running main.py in {} mode with lang: {}".format(args.mode, args.lang))

    # load the data from the specific path

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    init_token = tokenizer.cls_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

    max_input_length = tokenizer.max_model_input_sizes['bert-base-multilingual-cased']
    
    #text_preprocessor = functools.partial(cut_and_convert_to_id, tokenizer = tokenizer,max_input_length = max_input_length)
    #tag_preprocessor = functools.partial(cut_to_max_length, max_input_length = max_input_length)

    # TEXT = data.Field(use_vocab = False,
    #               lower = True,
    #               preprocessing = transform_text,
    #               init_token = init_token_idx,
    #               pad_token = pad_token_idx,
    #               unk_token = unk_token_idx)

    UD_TAGS = data.Field(unk_token = None,
                     init_token = '<pad>',
                     preprocessing = preprocess_tag)

    train_data, valid_data, test_data = UDPOS(
        os.path.join('data', args.lang),
        split=('train', 'valid', 'test'),
    )

    train_tags = [label for (line, label) in train_data + valid_data + test_data] 

    UD_TAGS.build_vocab(train_tags)

    print("UD_TAG vocabulary")
    print(UD_TAGS.vocab.stoi)


    OUTPUT_DIM = len(UD_TAGS.vocab)
    BATCH_SIZE = 32
    DROPOUT = 0.25

    #train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size = BATCH_SIZE,device = device)
    TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
    print(UD_TAGS.pad_token)



    def collate_batch(batch):
        tag_list, text_list = [], []
        for (line, label) in batch:
            text_list.append(torch.tensor(transform_text(line, tokenizer, max_input_length), device=device))
            tag_list.append(torch.tensor(transform_tag(label, max_input_length, UD_TAGS), device=device))
        return (
            pad_sequence(text_list, padding_value=TAG_PAD_IDX),
            pad_sequence(tag_list, padding_value=TAG_PAD_IDX)
        )

    train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], collate_fn=collate_batch,shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=params['batch_size'],collate_fn=collate_batch,shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=params['batch_size'],collate_fn=collate_batch,shuffle=False)   
    bert = BertModel.from_pretrained('bert-base-multilingual-cased')
    model = BERTPoSTagger(bert, OUTPUT_DIM, DROPOUT)

    LEARNING_RATE = 5e-5
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {n_param} trainable parameters")
    N_EPOCHS = params['max_epoch']
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, TAG_PAD_IDX)
        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, TAG_PAD_IDX)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"saved_models/{args.model_name}.pt")

        print(f"Epoch: {epoch+1:02} "
                  f"| Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} "
                  f"| Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} "
                  f"|  Val. Acc: {valid_acc*100:.2f}%")


    test_loss, test_acc = evaluate(model, test_dataloader, criterion,TAG_PAD_IDX)
    print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%")

def transform_text(tokens, tokenizer, max_input_length):
    tokens = tokens[:max_input_length-1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens

def transform_tag(tokens, max_input_length,UD_TAGS):
    tokens = tokens[:max_input_length-1]
    tokens = [UD_TAGS.vocab.stoi[token] for token in tokens]
    return tokens

def preprocess_tag(tokens, max_input_length):
    tokens = tokens[:max_input_length-1]
    return tokens

def train(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        text = batch[0]
        tags = batch[1]
        print(text)
        print(tags)
        print(text.shape)
        print(tags.shape)
        optimizer.zero_grad()
        
        #text = [sent len, batch size]
        
        predictions = model(text)
    
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        
        loss = criterion(predictions, tags)
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def tag_percentage(tag_counts):
    total_count = sum([count for tag, count in tag_counts])
    tag_counts_percentages = [
        (tag, count, count / total_count) for tag, count in tag_counts
    ]
    return tag_counts_percentages


def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)

def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text = batch[0]
            tags = batch[1]
            predictions = model(text)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            loss = criterion(predictions, tags)
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    main()
