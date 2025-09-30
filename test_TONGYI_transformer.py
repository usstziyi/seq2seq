import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader, random_split
import random
from collections import Counter
import numpy as np
import re
import os
import urllib.request
import zipfile
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class Vocabulary:
    def __init__(self, max_vocab_size=10000):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.max_vocab_size = max_vocab_size
        
    def build_vocab(self, sentences, min_freq=2):
        # 添加特殊标记
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        for i, token in enumerate(special_tokens):
            self.word2idx[token] = i
        
        # 统计词频
        word_count = Counter()
        for sentence in sentences:
            words = self.tokenize(sentence)
            word_count.update(words)
        
        # 只保留高频词汇
        common_words = word_count.most_common(self.max_vocab_size - len(special_tokens))
        filtered_words = [(word, count) for word, count in common_words if count >= min_freq]
        
        for word, _ in filtered_words:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        
        self.vocab_size = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def tokenize(self, sentence):
        # 简单的分词方法
        sentence = sentence.lower().strip()
        sentence = re.sub(r"[^a-zA-ZÀ-ÿ\s]", '', sentence)
        return sentence.split()
    
    def encode(self, sentence):
        words = self.tokenize(sentence)
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
    
    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices if idx not in [0, 2, 3]])  # 排除PAD, SOS, EOS

class TranslationDataset(Dataset):
    def __init__(self, english_sentences, french_sentences, eng_vocab, fr_vocab, max_length=50):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences
        self.eng_vocab = eng_vocab
        self.fr_vocab = fr_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self, idx):
        eng_sentence = self.english_sentences[idx]
        fr_sentence = self.french_sentences[idx]
        
        # 编码句子
        eng_indices = self.eng_vocab.encode(eng_sentence)
        fr_indices = self.fr_vocab.encode(fr_sentence)
        
        # 添加开始和结束标记
        fr_input = [self.fr_vocab.word2idx['<SOS>']] + fr_indices
        fr_target = fr_indices + [self.fr_vocab.word2idx['<EOS>']]
        
        # 截断或填充
        eng_indices = self._pad_sequence(eng_indices, self.max_length)
        fr_input = self._pad_sequence(fr_input, self.max_length)
        fr_target = self._pad_sequence(fr_target, self.max_length)
        
        return (torch.tensor(eng_indices), 
                torch.tensor(fr_input), 
                torch.tensor(fr_target))
    
    def _pad_sequence(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [0] * (max_length - len(sequence))  # 0是<PAD>的索引

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerTranslator, self).__init__()
        
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
            batch_first=False  # 注意：PyTorch Transformer默认batch_first=False
        )
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        # 转置以适应Transformer的输入格式 (seq_len, batch_size)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        # 嵌入和位置编码
        src_embed = self.dropout(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_embed = self.dropout(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        src_embed = self.pos_encoder(src_embed)
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # 创建mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        
        # Transformer前向传播
        output = self.transformer(src_embed, tgt_embed, tgt_mask=tgt_mask)
        
        # 输出层
        output = self.fc_out(output)
        
        return output.transpose(0, 1)  # 转回(batch_size, seq_len, vocab_size)

def download_dataset(url, filename):
    """下载数据集"""
    if not os.path.exists(filename):
        print(f"正在下载 {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("下载完成!")
    else:
        print(f"{filename} 已存在")

def extract_dataset(zip_filename, extract_to='.'):
    """解压数据集"""
    if zip_filename.endswith('.zip'):
        print(f"正在解压 {zip_filename}...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("解压完成!")

def load_tatoeba_data(data_file, max_samples=None):
    """加载Tatoeba数据集"""
    english_sentences = []
    french_sentences = []
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if max_samples:
            lines = lines[:max_samples]
            
        for line in tqdm(lines, desc="Processing data"):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                english_sentences.append(parts[0])
                french_sentences.append(parts[1])
                
    except FileNotFoundError:
        print(f"未找到文件 {data_file}，请确保数据集已正确下载")
        return [], []
    
    return english_sentences, french_sentences

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, validation_loader=None):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        
        for batch_idx, (src, tgt_input, tgt_output) in enumerate(progress_bar):
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            
            # 忽略PAD标记的损失
            loss = criterion(output, tgt_output)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # 验证
        if validation_loader:
            val_loss = evaluate_model(model, validation_loader, criterion, device)
            print(f'Validation Loss: {val_loss:.4f}')

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt_input, tgt_output in data_loader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)
            
            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(data_loader)

def translate_sentence(model, sentence, eng_vocab, fr_vocab, device, max_length=50):
    model.eval()
    
    with torch.no_grad():
        # 编码输入句子
        src_indices = eng_vocab.encode(sentence)
        if len(src_indices) == 0:
            return "无法处理的输入"
            
        src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
        
        # 初始化目标序列
        tgt_indices = [fr_vocab.word2idx['<SOS>']]
        
        for i in range(max_length):
            tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
            
            output = model(src_tensor, tgt_tensor)
            next_token = output[0, -1, :].argmax(dim=-1).item()
            
            tgt_indices.append(next_token)
            
            if next_token == fr_vocab.word2idx['<EOS>']:
                break
        
        # 解码输出
        translated_words = [fr_vocab.idx2word[idx] for idx in tgt_indices[1:-1]]  # 移除SOS和EOS
        return ' '.join(translated_words)

def main():
    # 下载Tatoeba英法翻译数据集
    dataset_url = "http://www.manythings.org/anki/fra-eng.zip"
    zip_filename = "fra-eng.zip"
    data_dir = "fra-eng"
    data_file = os.path.join(data_dir, "fra.txt")
    
    # 下载并解压数据集
    download_dataset(dataset_url, zip_filename)
    extract_dataset(zip_filename, data_dir)
    
    # 加载数据集
    print("正在加载数据集...")
    english_sentences, french_sentences = load_tatoeba_data(data_file, max_samples=50000)  # 限制样本数量以加快训练
    
    if len(english_sentences) == 0:
        print("数据加载失败，请检查数据文件路径")
        return
    
    print(f"成功加载 {len(english_sentences)} 对翻译句子")
    print("示例数据:")
    for i in range(3):
        print(f"English: {english_sentences[i]}")
        print(f"French:  {french_sentences[i]}")
        print()
    
    # 创建词汇表
    eng_vocab = Vocabulary(max_vocab_size=15000)
    fr_vocab = Vocabulary(max_vocab_size=15000)
    
    print("正在构建词汇表...")
    eng_vocab.build_vocab(english_sentences, min_freq=2)
    fr_vocab.build_vocab(french_sentences, min_freq=2)
    
    print(f"English vocabulary size: {eng_vocab.vocab_size}")
    print(f"French vocabulary size: {fr_vocab.vocab_size}")
    
    # 数据集划分
    dataset = TranslationDataset(english_sentences, french_sentences, eng_vocab, fr_vocab, max_length=30)
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = TransformerTranslator(
        src_vocab_size=eng_vocab.vocab_size,
        tgt_vocab_size=fr_vocab.vocab_size,
        d_model=256,  # 较小的模型便于快速训练
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标记
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # 训练模型
    print("开始训练...")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=20, validation_loader=val_loader)
    
    # 测试翻译
    print("\n测试翻译:")
    test_sentences = [
        "Hello world",
        "Good morning",
        "How are you",
        "Thank you very much",
        "Where is the bathroom"
    ]
    
    for sentence in test_sentences:
        translated = translate_sentence(model, sentence, eng_vocab, fr_vocab, device)
        print(f"English: {sentence}")
        print(f"French:  {translated}")
        print()
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'eng_vocab': eng_vocab,
        'fr_vocab': fr_vocab,
        'model_config': {
            'src_vocab_size': eng_vocab.vocab_size,
            'tgt_vocab_size': fr_vocab.vocab_size,
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 512
        }
    }, 'transformer_en_fr.pth')
    
    print("模型已保存为 transformer_en_fr.pth")

if __name__ == "__main__":
    main()
