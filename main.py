import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torchaudio.functional import rnnt_loss
# from RNNT import RNNTModel
import pickle
import ipdb
from tqdm import tqdm


class SpeechDataset(Dataset):
    def __init__(self, data_dir, tokenizer, sample_rate=16000, n_mels=80):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.data = self._load_data()

    def _load_data(self):
        data = []
        # ipdb.set_trace()
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    audio_path = os.path.join(root, file)
                    transcript_path = audio_path + ".trn"
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        transcript = f.readlines()[0].strip()
                    data.append((audio_path, transcript))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, transcript = self.data[idx]
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        # 梅尔频谱图 tensor(n_mels, audio_length)
        mel_spec = T.MelSpectrogram(sample_rate=self.sample_rate, n_mels=self.n_mels)(waveform).squeeze(0)
        tokenized_transcript = torch.tensor(self.tokenizer.encode(transcript), dtype=torch.long)
        return mel_spec.T, tokenized_transcript


class RNNTModel(nn.Module):
    def __init__(self, input_dim, vocab_size, encoder_hidden, pred_hidden):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, encoder_hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.pred_net = nn.Embedding(vocab_size, pred_hidden)
        self.pred_lstm = nn.LSTM(pred_hidden, pred_hidden, num_layers=1, batch_first=True)
        self.joint_net = nn.Sequential(
            nn.Linear(encoder_hidden * 2 + pred_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_size))
    
    def forward(self, auto_features, targets):
        enc_out, _ = self.encoder(auto_features)
        pred_emb = self.pred_net(targets)
        pred_out, _ = self.pred_lstm(pred_emb)
        # ipdb.set_trace()
        enc_out_exp = enc_out.unsqueeze(2)    # [B, T, 1, H]
        pred_out_exp = pred_out.unsqueeze(1)    # [B, 1, U, H]
        joint_input = torch.cat((enc_out_exp.expand(-1, -1, pred_out.size(1), -1),
                                 pred_out_exp.expand(-1, enc_out.size(1), -1, -1)), dim=-1)
        logits = self.joint_net(joint_input)  # [B, T, U, V]
        return logits

class CharTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def encode(self, text):
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char])

def collate_fn(batch):
    audio_features = [item[0] for item in batch]
    transcripts = [item[1] for item in batch]
    audio_features = pad_sequence(audio_features, batch_first=True)
    transcripts = pad_sequence(transcripts, batch_first=True)
    return audio_features, transcripts


DATA_DIR = r"/root/autodl-tmp/data_thchs30/data"
with open(r"/root/autodl-tmp/char_set.pkl", "rb") as file:
    char_set = pickle.load(file)

VOCAB = ['<blank>', '<sos>', '<eos>'] + list(char_set)  # 词汇表
INPUT_DIM = 80
ENCODER_HIDDEN = 256
PRED_HIDDEN = 128
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-3

# Tokenizer 和数据集
tokenizer = CharTokenizer(VOCAB)
train_dataset = SpeechDataset(DATA_DIR, tokenizer)
# ipdb.set_trace()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNTModel(INPUT_DIM, len(VOCAB), ENCODER_HIDDEN, PRED_HIDDEN).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
for epoch in tqdm(range(EPOCHS)):
    model.train()
    epoch_loss = 0
    for audio_features, targets in train_loader:
        # ipdb.set_trace()
        audio_features = audio_features.to(device)
        targets = targets.to(torch.int32)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 模型推理
        logits = model(audio_features, targets)  # [B, T, U, V]

        # 计算长度
        logit_lengths = torch.tensor([audio_features.size(1)] * audio_features.size(0), dtype=torch.int32).to(device)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.int32).to(device)

        # 计算损失
        # ipdb.set_trace()
        loss = rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0)

        # 反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader)}")

# 模型保存
torch.save(model.state_dict(), "rnnt_thchs30.pth")
