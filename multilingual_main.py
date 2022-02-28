#REFERENCES Adapted from https://github.com/bentrevett/pytorch-pos-tagging

from collections import Counter
import torch
from torchtext.legacy import data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    print("Running multilingual_main.py in {} mode with lang: {}".format(args.mode, args.lang))

    # load the data from the specific path
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    max_input_length = tokenizer.max_model_input_sizes['bert-base-multilingual-cased']
    UD_TAGS = data.Field(unk_token = None,
                     init_token = '<pad>',
                     preprocessing = preprocess_tag)

    train_data, valid_data, test_data = UDPOS(
        os.path.join('data', args.lang),
        split=('train', 'valid', 'test'),
    )

    train_tags = [label for (line, label) in train_data + valid_data + test_data] 
    UD_TAGS.build_vocab(train_tags)
    # print("UD_TAG vocabulary")
    # print(UD_TAGS.vocab.stoi)


    OUTPUT_DIM = len(UD_TAGS.vocab)
    BATCH_SIZE = params["batch_size"]
    DROPOUT = params["dropout"]
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

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_batch,shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE,collate_fn=collate_batch,shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,collate_fn=collate_batch,shuffle=False)   
    bert = BertModel.from_pretrained('bert-base-multilingual-cased')
    model = BERTPoSTagger(bert, OUTPUT_DIM, DROPOUT)

    LEARNING_RATE = params["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)

    if args.mode == "train":
        n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The model has {n_param} trainable parameters")
        N_EPOCHS = params['max_epoch']
        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, TAG_PAD_IDX)
            valid_loss, valid_acc, outputs = evaluate(model, valid_dataloader, criterion, TAG_PAD_IDX)
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

    try:
        model.load_state_dict(
            torch.load(f"saved_models/{args.model_name}.pt",
                       map_location=device)
        )
    except OSError:
        print(
            f"Model file `saved_models/{args.model_name}.pt` doesn't exist."
            "You need to train the model by running this code in train mode."
            "Run python main.py --help for more instructions"
        )
        return
    
    test_loss, test_acc, outputs = evaluate(model, test_dataloader, criterion,TAG_PAD_IDX)
    print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%")

    output_path = os.path.join('model_outputs', f'{args.lang}.conll')
    dump_output(test_data, outputs, UD_TAGS, output_path)





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
        optimizer.zero_grad()
        predictions = model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
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
    max_preds = preds.argmax(dim = 1, keepdim = True) 
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)

def evaluate(model, iterator, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in iterator:
            text = batch[0]
            tags = batch[1]
            predictions = model(text)
            outputs += [
                pred[:length].cpu()
                for pred, length in zip(
                        predictions.argmax(-1).transpose(0, 1),
                        (tags != tag_pad_idx).long().sum(0)
                )
            ]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            loss = criterion(predictions, tags)
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), outputs


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def tag_percentage(tag_counts):
    total_count = sum([count for tag, count in tag_counts])
    tag_counts_percentages = [(tag, count, count/total_count) for tag, count in tag_counts]
    return tag_counts_percentages

def get_statistics(UD_TAGS):
    print(UD_TAGS.vocab.freqs.most_common())
    print("Tag\t\tCount\t\tPercentage\n")
    for tag, count, percent in tag_percentage(UD_TAGS.vocab.freqs.most_common()):
        print(f"{tag}\t\t{count}\t\t{percent*100:4.1f}%")

def dump_output(data, outputs, vocab_tag, output_path):
    assert len(data) == len(outputs)
    with open(output_path, 'w') as f:
        for (line, _), output in zip(data, outputs):
            for token, tag in zip(line, output):
                f.write(f'{token}\t{vocab_tag.vocab.itos[tag]}\n')
            f.write('\n')




if __name__ == "__main__":
    main()
