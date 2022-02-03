from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from bpemb import BPEmb

class BiLSTMPOSTaggerMultilingual(nn.Module):
  

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
        lang):

        super().__init__()

        # self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        # self.model = BertModel.from_pretrained("bert-base-multilingual-uncased")

        self.lang = lang
        self.embedding_dim = embedding_dim
        self.bpe_embedding = BPEmb(lang=lang,dim=embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding_text):

        # text = [sent len, batch size]

        # pass text through embedding layer

        # multilingual_embedding = 

        # encoded_input = self.tokenizer(text, return_tensors='pt')
        
        # output = self.model(**encoded_input)

        # print(output.shape
        
        # print(output[0])
         embedded = self.dropout(embedding_text)
         outputs, (hidden, cell) = self.lstm(embedded)
         predictions = self.fc(self.dropout(outputs))
         return predictions



        #embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pass embeddings into LSTM
        #outputs, (hidden, cell) = self.lstm(embedded)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        #predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]

        


    # def get_multilingual_embedding():
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    #     model = BertModel.from_pretrained("bert-base-multilingual-uncased")
    #     text = "Replace me by any text you'd like."
    #     encoded_input = tokenizer(text, return_tensors='pt')
        