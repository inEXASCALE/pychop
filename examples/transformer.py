import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import warnings

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage",
    category=UserWarning
)



torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = [
    ("I am a student", "Je suis étudiant"),
    ("We are happy", "Nous sommes heureux"),
    ("They read books", "Ils lisent des livres"),
    ("She loves music", "Elle aime la musique"),
]

# Vocabulary creation
def build_vocab(sentences):
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    idx = len(vocab)
    for sent in sentences:
        for word in sent.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

# Build vocab for source (English) and target (French)
src_sentences, tgt_sentences = zip(*data)
src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# Reverse vocab for decoding
src_idx_to_word = {idx: word for word, idx in src_vocab.items()}
tgt_idx_to_word = {idx: word for word, idx in tgt_vocab.items()}

# Convert sentences to indices
def sentence_to_indices(sentence, vocab):
    return [vocab["<SOS>"]] + [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()] + [vocab["<EOS>"]]

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.src_data = []
        self.tgt_data = []
        for src, tgt in data:
            self.src_data.append(sentence_to_indices(src, src_vocab))
            self.tgt_data.append(sentence_to_indices(tgt, tgt_vocab))
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long), torch.tensor(self.tgt_data[idx], dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=src_vocab["<PAD>"], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab["<PAD>"], batch_first=True)
    return src_batch, tgt_batch

dataset = TranslationDataset(data, src_vocab, tgt_vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=3, 
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model


        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return self.fc_out(output)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_key_padding_mask = (src == src_vocab["<PAD>"]).to(device)
            tgt_key_padding_mask = (tgt_input == tgt_vocab["<PAD>"]).to(device)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            optimizer.zero_grad()
            output = model(
                src,
                tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}')

def translate(model, src_sentence, src_vocab, tgt_vocab, src_idx_to_word, tgt_idx_to_word, max_len=50):
    model.eval()
    src = torch.tensor(sentence_to_indices(src_sentence, src_vocab), dtype=torch.long).unsqueeze(0).to(device)
    tgt = torch.tensor([tgt_vocab["<SOS>"]], dtype=torch.long).unsqueeze(0).to(device)
    
    src_key_padding_mask = (src == src_vocab["<PAD>"]).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            output = model(
                src,
                tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask
            )
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            if next_token.item() == tgt_vocab["<EOS>"]:
                break
    
    translated = [tgt_idx_to_word[idx.item()] for idx in tgt[0]]
    return " ".join(translated[1:-1])  # Exclude <SOS> and <EOS>

d_model = 512
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 2048
dropout = 0.1

model = TransformerModel(
    src_vocab_size,
    tgt_vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

num_epochs = 50
print("Training started...")
train(model, dataloader, criterion, optimizer, num_epochs)

test_sentence = "I am a student"
translated = translate(model, test_sentence, src_vocab, tgt_vocab, src_idx_to_word, tgt_idx_to_word)
print(f"\nSource: {test_sentence}")
print(f"Translated: {translated}")