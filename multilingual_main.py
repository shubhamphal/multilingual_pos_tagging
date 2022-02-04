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

    TEXT = data.Field(use_vocab = False,
                  lower = True,
                  preprocessing = transform_text,
                  init_token = init_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

    UD_TAGS = data.Field(unk_token = None,
                     init_token = '<pad>',
                     preprocessing = transform_tag)

    train_data, valid_data, test_data = UDPOS(
        os.path.join('data', args.lang),
        split=('train', 'valid', 'test'),
    )

    UD_TAGS.build_vocab(train_data)

    OUTPUT_DIM = len(UD_TAGS.vocab)
    BATCH_SIZE = 32
    DROPOUT = 0.25

    #train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size = BATCH_SIZE,device = device)
   
    TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

    def collate_batch(batch):
        tag_list, text_list = [], []
        for (line, label) in batch:
            text_list.append(torch.tensor(transform_text(line, tokenizer, max_input_length), device=device))
            tag_list.append(torch.tensor(transform_tag(label, max_input_length), device=device))
        return (
            pad_sequence(text_list, padding_value=TAG_PAD_IDX),
            pad_sequence(tag_list, padding_value=TAG_PAD_IDX)
        )



    train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=params['batch_size'],shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=params['batch_size'],shuffle=False)


   
    bert = BertModel.from_pretrained('bert-base-multilingual-cased')
    model = BERTPoSTagger(bert, OUTPUT_DIM, DROPOUT)

    LEARNING_RATE = 5e-5
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    
    
    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {n_param} trainable parameters")

    N_EPOCHS = 10
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

def transform_tag(tokens, max_input_length):
    tokens = tokens[:max_input_length-1]
    return tokens


#     #code to count the number of tokens in each language
#     # all_tokens = [line for line, label in train_data]
#     # token_list = []
#     # for line in all_tokens:
#     #     for token in line:
#     #         token_list.append(token)
#     # print(len(token_list))

#     #building the vocabulary for both text and the labels
#     vocab_text = torchtext.vocab.build_vocab_from_iterator(
#         (line for line, label in train_data), min_freq=params['min_freq'],
#         specials=['<unk>', '<PAD>']
#     )
#     vocab_text.set_default_index(vocab_text['<unk>'])
#     vocab_tag = torchtext.vocab.build_vocab_from_iterator(
#         (label for line, label in train_data),
#         specials=['<unk>', '<PAD>']
#     )
#     vocab_tag.set_default_index(vocab_tag['<unk>'])

#     # print the statistics of the data
#     if args.mode == "train":
#         print(f"Unique tokens in TEXT vocabulary: {len(vocab_text)}")
#         print(f"Unique tokens in UD_TAG vocabulary: {len(vocab_tag)}")
#         print()
#         print(f"Number of training examples: {len(train_data)}")
#         print(f"Number of validation examples: {len(valid_data)}")
#     print(f"Number of testing examples: {len(test_data)}")

#     # functions that convert words/tags to indices
#     def transform_text(x):
#         return [vocab_text[token] for token in x]

#     def transform_tag(x):
#         return [vocab_tag[tag] for tag in x]

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # function that pack (text, label) pairs into a batch
#     def collate_batch(batch):
#         tag_list, text_list = [], []
#         for (line, label) in batch:
#             text_list.append(torch.tensor(transform_text(line), device=device))
#             tag_list.append(torch.tensor(transform_tag(label), device=device))
#         return (
#             pad_sequence(text_list, padding_value=vocab_text['<PAD>']),
#             pad_sequence(tag_list, padding_value=vocab_tag['<PAD>'])
#         )

#     # iterators that generate batch with `collate_batch`
#     train_dataloader = DataLoader(
#         train_data, batch_size=params['batch_size'],
#         shuffle=True, collate_fn=collate_batch
#     )
#     valid_dataloader = DataLoader(
#         valid_data, batch_size=params['batch_size'],
#         shuffle=False, collate_fn=collate_batch
#     )
#     test_dataloader = DataLoader(
#         test_data, batch_size=params['batch_size'],
#         shuffle=False, collate_fn=collate_batch
#     )

#     # initialize the model
#     model = BiLSTMPOSTagger(
#         input_dim=len(vocab_text),
#         embedding_dim=params["embedding_dim"],
#         hidden_dim=params["hidden_dim"],
#         output_dim=len(vocab_tag),
#         n_layers=params["n_layers"],
#         bidirectional=params["bidirectional"],
#         dropout=params["dropout"],
#         pad_idx=vocab_text['<PAD>'],
#     ).to(device)

#     # print the number of parameters
#     n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"The model has {n_param} trainable parameters")

#     # set up the loss function
#     criterion = nn.CrossEntropyLoss(ignore_index=vocab_tag['<PAD>'])
#     criterion = criterion.to(device)

#     if args.mode == "train":
#         # initialize the parameters
#         def init_weights(m):
#             for name, param in m.named_parameters():
#                 nn.init.normal_(param.data, mean=0, std=0.1)
#         model.apply(init_weights)
#         # initialize the embedding for <PAD> with zero
#         model.embedding.weight.data[vocab_text['<PAD>']] = \
#             torch.zeros(params["embedding_dim"])

#         # set up the optimizer
#         optimizer = optim.Adam(model.parameters())

#         # training loop
#         best_valid_loss = float("inf")
#         for epoch in range(params['max_epoch']):
#             start_time = time.time()
#             train_loss, train_acc = train(
#                 model,
#                 train_dataloader,
#                 optimizer,
#                 criterion,
#                 vocab_tag['<PAD>'],
#                 vocab_tag['<UNK>'],
#             )
#             valid_loss, valid_acc = evaluate(
#                 model, valid_dataloader, criterion,
#                 vocab_tag['<PAD>'], vocab_tag['<UNK>']
#             )
#             end_time = time.time()
#             epoch_mins, epoch_secs = epoch_time(start_time, end_time)

#             # save the model if valid_loss is better
#             if valid_loss < best_valid_loss:
#                 best_valid_loss = valid_loss
#                 torch.save(
#                     model.state_dict(), f"saved_models/{args.model_name}.pt"
#                 )

#             print(f"Epoch: {epoch+1:02} "
#                   f"| Epoch Time: {epoch_mins}m {epoch_secs}s")
#             print(f"\tTrain Loss: {train_loss:.3f} "
#                   f"| Train Acc: {train_acc*100:.2f}%")
#             print(f"\t Val. Loss: {valid_loss:.3f} "
#                   f"|  Val. Acc: {valid_acc*100:.2f}%")

#     # load the trained model
#     try:
#         model.load_state_dict(
#             torch.load(f"saved_models/{args.model_name}.pt",
#                        map_location=device)
#         )
#     except OSError:
#         print(
#             f"Model file `saved_models/{args.model_name}.pt` doesn't exist."
#             "You need to train the model by running this code in train mode."
#             "Run python main.py --help for more instructions"
#         )
#         return

#     test_loss, test_acc = evaluate(
#         model, test_dataloader, criterion, vocab_tag['<PAD>'], vocab_tag['<UNK>']
#     )
#     print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%")

def train(model, iterator, optimizer, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        text = batch.text
        tags = batch.udtags
                
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


# def train(model, iterator, optimizer, criterion, tag_pad_idx, tag_unk_idx):

#     epoch_loss = 0
#     epoch_correct = 0
#     epoch_n_label = 0

#     model.train()

#     for batch in iterator:

#         text = batch[0]
#         tags = batch[1]

#         optimizer.zero_grad()

#         # text = [sent len, batch size]

#         predictions = model(text)

#         # predictions = [sent len, batch size, output dim]
#         # tags = [sent len, batch size]

#         predictions = predictions.view(-1, predictions.shape[-1])
#         tags = tags.view(-1)

#         # predictions = [sent len * batch size, output dim]
#         # tags = [sent len * batch size]

#         loss = criterion(predictions, tags)

#         correct, n_labels = categorical_accuracy(
#             predictions, tags, tag_pad_idx, tag_unk_idx
#         )

#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()
#         epoch_correct += correct.item()
#         epoch_n_label += n_labels

#     return epoch_loss / len(iterator), epoch_correct / epoch_n_label


def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text = batch.text
            tags = batch.udtags
            
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
