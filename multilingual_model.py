from turtle import Shape
import torch
import torch.nn as nn

class BERTPoSTagger(nn.Module):
    def __init__(self,
                 bert,
                 output_dim, 
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
  
        #text = [sent len, batch size]
        text = text.permute(1, 0)
        #text = [batch size, sent len]
        embedded = self.dropout(self.bert(text)[0])
        
        #embedded = [batch size, seq len, emb dim]
        embedded = embedded.permute(1, 0, 2)
        print("Shape")
        print(embedded.shape)

        #embedded = [sent len, batch size, emb dim]
        
        predictions = self.fc(self.dropout(embedded))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions