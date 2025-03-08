import warnings
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage",
    category=UserWarning
)

import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data Preparation
# Load the IWSLT'14 dataset (English-German)
dataset = load_dataset("iwslt2017", "iwslt2017-en-de")
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Tokenizers for English and German
en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")

# Function to build vocabulary
def build_vocab(data, tokenizer, lang="en", min_freq=2):
    counter = Counter()
    for example in data:
        if lang == "en":
            tokens = tokenizer(example["translation"]["en"])
        else:
            tokens = tokenizer(example["translation"]["de"])
        counter.update(tokens)
    
    # Add special tokens
    specials = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    vocab = Vocab(counter, specials=specials, min_freq=min_freq)
    vocab.set_default_index(vocab["<UNK>"])  # Set default index for unknown tokens
    return vocab

# Build vocabularies for English and German
src_vocab = build_vocab(train_data, en_tokenizer, lang="en", min_freq=2)
tgt_vocab = build_vocab(train_data, de_tokenizer, lang="de", min_freq=2)
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# Reverse vocab for decoding
src_idx_to_word = src_vocab.get_itos()
tgt_idx_to_word = tgt_vocab.get_itos()

# Convert sentences to indices
def sentence_to_indices(sentence, vocab, tokenizer):
    tokens = tokenizer(sentence)
    indices = [vocab["<SOS>"]] + [vocab[token] for token in tokens] + [vocab["<EOS>"]]
    return indices

# Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        self.src_data = []
        self.tgt_data = []
        for example in data:
            src_sentence = example["translation"]["en"]
            tgt_sentence = example["translation"]["de"]
            src_indices = sentence_to_indices(src_sentence, src_vocab, src_tokenizer)
            tgt_indices = sentence_to_indices(tgt_sentence, tgt_vocab, tgt_tokenizer)
            self.src_data.append(src_indices)
            self.tgt_data.append(tgt_indices)
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long), torch.tensor(self.tgt_data[idx], dtype=torch.long)

# Collate function for padding
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=src_vocab["<PAD>"], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab["<PAD>"], batch_first=True)
    return src_batch, tgt_batch

# Create datasets and dataloaders
train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, en_tokenizer, de_tokenizer)
val_dataset = TranslationDataset(val_data, src_vocab, tgt_vocab, en_tokenizer, de_tokenizer)
test_dataset = TranslationDataset(test_data, src_vocab, tgt_vocab, en_tokenizer, de_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 2. Positional Encoding
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

# 3. Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
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
        
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(output)

# Generate square subsequent mask for target (decoder self-attention)
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

# 4. Training and Validation Loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        for src, tgt in train_loader:
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
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                src_key_padding_mask = (src == src_vocab["<PAD>"]).to(device)
                tgt_key_padding_mask = (tgt_input == tgt_vocab["<PAD>"]).to(device)
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                
                output = model(
                    src,
                    tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )
                loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# 5. Inference (Greedy Decoding)
def translate(model, src_sentence, src_vocab, tgt_vocab, src_tokenizer, tgt_idx_to_word, max_len=50):
    model.eval()
    src_tokens = src_tokenizer(src_sentence)
    src_indices = [src_vocab["<SOS>"]] + [src_vocab[token] for token in src_tokens] + [src_vocab["<EOS>"]]
    src = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
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

# 6. Run Training and Inference
# Model hyperparameters
d_model = 512
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 2048
dropout = 0.1

# Initialize model, loss, and optimizer
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

# Train the model
num_epochs = 10
print("Training started...")
train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Test translation
test_sentence = "I am a student."
translated = translate(model, test_sentence, src_vocab, tgt_vocab, en_tokenizer, tgt_idx_to_word)
print(f"\nSource: {test_sentence}")
print(f"Translated: {translated}")