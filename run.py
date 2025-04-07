"""
- Copyright (c) 2025 DuYu (No.202103180009, qluduyu09@163.com), Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences).
- 基于Transformer的汉语拼音序列转汉字序列模型 训练与测试代码
- 文件名：run.py
"""
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

warnings.filterwarnings("ignore")  # 全局禁用警告信息，开发时可去除

# 设置随机种子保证可重复性
torch.manual_seed(525200)
np.random.seed(40004004)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. 数据读取与预处理
class PinyinHanziDataset(Dataset):
    def __init__(self, csv_file, max_length=15):
        self.data = pd.read_csv(csv_file, header=None, names=['hanzi', 'pinyin'])
        self.max_length = max_length

        # 构建词汇表
        self._build_vocab()

    def _tokenize_hanzi(self, s):
        """将文本分割为汉字、英文单词和标点符号的混合token"""
        pattern = re.compile(
            r'([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]|[a-zA-Z.,!?;:\'"]+|\d+|\s)'
        )

        tokens = []
        for token in pattern.finditer(s):
            if token.group().strip():  # 忽略纯空格
                tokens.append(token.group())

        return tokens

    def _build_vocab(self):
        # 处理汉字词汇表
        hanzi_counter = Counter()
        pinyin_counter = Counter()

        for _, row in self.data.iterrows():
            # 使用新的tokenize方法处理汉字
            hanzi_tokens = self._tokenize_hanzi(row['hanzi'])
            hanzi_counter.update(hanzi_tokens)

            # 拼音处理：按空格分割
            pinyin_tokens = row['pinyin'].split()
            pinyin_counter.update(pinyin_tokens)

        # 添加特殊token
        self.hanzi_vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + [char for char, _ in hanzi_counter.most_common()]
        self.pinyin_vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + [pinyin for pinyin, _ in
                                                                    pinyin_counter.most_common()]

        # 创建token到id的映射
        self.hanzi2idx = {char: idx for idx, char in enumerate(self.hanzi_vocab)}
        self.idx2hanzi = {idx: char for idx, char in enumerate(self.hanzi_vocab)}
        self.pinyin2idx = {pinyin: idx for idx, pinyin in enumerate(self.pinyin_vocab)}
        self.idx2pinyin = {idx: pinyin for idx, pinyin in enumerate(self.pinyin_vocab)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hanzi_seq = self.data.iloc[idx]['hanzi']
        pinyin_seq = self.data.iloc[idx]['pinyin']

        # 将汉字序列转换为token id序列
        hanzi_tokens = ['<sos>'] + self._tokenize_hanzi(hanzi_seq) + ['<eos>']
        hanzi_ids = [self.hanzi2idx.get(token, self.hanzi2idx['<unk>']) for token in hanzi_tokens]

        # 将拼音序列转换为token id序列
        pinyin_tokens = ['<sos>'] + pinyin_seq.split() + ['<eos>']
        pinyin_ids = [self.pinyin2idx.get(token, self.pinyin2idx['<unk>']) for token in pinyin_tokens]

        # 截断或填充序列
        hanzi_ids = hanzi_ids[:self.max_length]
        pinyin_ids = pinyin_ids[:self.max_length]

        hanzi_padding = [self.hanzi2idx['<pad>']] * (self.max_length - len(hanzi_ids))
        pinyin_padding = [self.pinyin2idx['<pad>']] * (self.max_length - len(pinyin_ids))

        hanzi_ids += hanzi_padding
        pinyin_ids += pinyin_padding

        return {
            'pinyin': torch.tensor(pinyin_ids, dtype=torch.long),
            'hanzi': torch.tensor(hanzi_ids, dtype=torch.long),
            'hanzi_input': torch.tensor(hanzi_ids[:-1], dtype=torch.long),
            'hanzi_target': torch.tensor(hanzi_ids[1:], dtype=torch.long)
        }


# 2. Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, pinyin_vocab_size, hanzi_vocab_size, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.075):
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # 拼音嵌入层
        self.pinyin_embedding = nn.Embedding(pinyin_vocab_size, d_model)
        # 汉字嵌入层
        self.hanzi_embedding = nn.Embedding(hanzi_vocab_size, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer模型
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # 输出层
        self.fc_out = nn.Linear(d_model, hanzi_vocab_size)

    def forward(self, pinyin, hanzi_input):
        # 嵌入层
        pinyin_embedded = self.pinyin_embedding(pinyin) * np.sqrt(self.d_model)
        hanzi_embedded = self.hanzi_embedding(hanzi_input) * np.sqrt(self.d_model)

        # 位置编码
        pinyin_embedded = self.positional_encoding(pinyin_embedded)
        hanzi_embedded = self.positional_encoding(hanzi_embedded)

        # 调整维度顺序：(seq_len, batch_size, d_model)
        pinyin_embedded = pinyin_embedded.permute(1, 0, 2)
        hanzi_embedded = hanzi_embedded.permute(1, 0, 2)

        # 创建mask
        src_mask = self._generate_square_subsequent_mask(pinyin_embedded.size(0)).to(device)
        tgt_mask = self._generate_square_subsequent_mask(hanzi_embedded.size(0)).to(device)

        # Transformer前向传播
        output = self.transformer(
            src=pinyin_embedded,
            tgt=hanzi_embedded,
            src_key_padding_mask=self._create_padding_mask(pinyin),
            tgt_key_padding_mask=self._create_padding_mask(hanzi_input),
            memory_key_padding_mask=self._create_padding_mask(pinyin),
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

        # 输出层，输出前将维度调整回(batch_size, seq_len, d_model)
        output = output.permute(1, 0, 2)
        output = self.fc_out(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _create_padding_mask(self, seq):
        return seq == 0  # 假设<pad>的id是0


# 3. 位置编码定义
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# 4. 建模（包装器定义）
class PinyinHanziTransformer:
    def __init__(self, model=None, dataset=None, config=None):
        self.model = model
        self.dataset = dataset
        self.config = config or {}

    def save(self, filepath):
        """保存整个模型、词汇表和配置到单个文件"""
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'hanzi_vocab': self.dataset.hanzi_vocab,
            'pinyin_vocab': self.dataset.pinyin_vocab,
            'hanzi2idx': self.dataset.hanzi2idx,
            'idx2hanzi': self.dataset.idx2hanzi,
            'pinyin2idx': self.dataset.pinyin2idx,
            'idx2pinyin': self.dataset.idx2pinyin,
            'max_length': self.dataset.max_length,
            'config': self.config
        }
        torch.save(save_data, filepath)

    @classmethod
    def load(cls, filepath, device='cpu'):
        """从文件加载整个模型"""
        save_data = torch.load(filepath, map_location=device)

        # 创建虚拟数据集对象以保存词汇表信息
        class DummyDataset:
            pass

        dataset = DummyDataset()
        dataset.hanzi_vocab = save_data['hanzi_vocab']
        dataset.pinyin_vocab = save_data['pinyin_vocab']
        dataset.hanzi2idx = save_data['hanzi2idx']
        dataset.idx2hanzi = save_data['idx2hanzi']
        dataset.pinyin2idx = save_data['pinyin2idx']
        dataset.idx2pinyin = save_data['idx2pinyin']
        dataset.max_length = save_data['max_length']

        # 初始化模型
        config = save_data['config']
        model = TransformerModel(
            pinyin_vocab_size=len(dataset.pinyin_vocab),
            hanzi_vocab_size=len(dataset.hanzi_vocab),
            **config
        ).to(device)
        model.load_state_dict(save_data['model_state_dict'])

        return cls(model=model, dataset=dataset, config=config)

    @staticmethod
    def top_k_sampling(logits, k=5, temperature=1.0):
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)  # shape: (1, vocab_size)

        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)  # shape: (1, k)

        # 从 top-k 中随机采样一个 index（在 top k 里的位置）
        sampled_index = torch.multinomial(topk_probs, num_samples=1)  # shape: (1, 1)

        # 找到对应的真正 vocab 索引
        next_token = torch.gather(topk_indices, dim=1, index=sampled_index)  # shape: (1, 1)

        # Instead of directly using .item(), ensure we're handling the tensor correctly
        return next_token.squeeze().item()  # .squeeze() to get rid of the extra dimension and then .item()

    def predict(self, pinyin_seq, max_length=None, k=3, temperature=1.0):
        """预测函数（使用top-k采样）"""
        self.model.eval()
        max_length = max_length or self.dataset.max_length

        # 拼音转ID
        pinyin_tokens = ['<sos>'] + pinyin_seq.split() + ['<eos>']
        pinyin_ids = [self.dataset.pinyin2idx.get(token, self.dataset.pinyin2idx['<unk>']) for token in pinyin_tokens]
        pinyin_ids = pinyin_ids[:max_length]
        pinyin_ids += [self.dataset.pinyin2idx['<pad>']] * (max_length - len(pinyin_ids))
        pinyin_tensor = torch.tensor(pinyin_ids, dtype=torch.long).unsqueeze(0).to(self.model.device)

        # 初始化汉字序列
        hanzi_ids = [self.dataset.hanzi2idx['<sos>']]

        for i in range(max_length - 1):
            hanzi_tensor = torch.tensor(hanzi_ids, dtype=torch.long).unsqueeze(0).to(self.model.device)

            with torch.no_grad():
                output = self.model(pinyin_tensor, hanzi_tensor)  # (1, seq_len, vocab_size)
                logits = output[:, -1, :]  # 取最后一个位置的logits，(1, vocab_size)

            # 使用top-k采样
            next_token = PinyinHanziTransformer.top_k_sampling(logits, k=k, temperature=temperature)
            hanzi_ids.append(next_token)

            if next_token == self.dataset.hanzi2idx['<eos>']:
                break

        # 转换为汉字序列
        hanzi_seq = [self.dataset.idx2hanzi[idx] for idx in hanzi_ids[1:-1]]  # 去掉<sos>和可能的<eos>
        return ''.join(hanzi_seq)

    # 在TransformerModel类中添加device属性
    @property
    def device(self):
        return next(self.parameters()).device

    TransformerModel.device = device


# 5. 训练函数定义
def train_model(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        pinyin = batch['pinyin'].to(device)
        hanzi_input = batch['hanzi_input'].to(device)
        hanzi_target = batch['hanzi_target'].to(device)

        # 前向传播
        output = model(pinyin, hanzi_input)

        # 计算损失
        loss = criterion(output.reshape(-1, output.size(-1)), hanzi_target.reshape(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / len(dataloader)


# 4. 评估函数定义
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pinyin = batch['pinyin'].to(device)
            hanzi_input = batch['hanzi_input'].to(device)
            hanzi_target = batch['hanzi_target'].to(device)

            output = model(pinyin, hanzi_input)
            loss = criterion(output.reshape(-1, output.size(-1)), hanzi_target.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


# 6. 模型训练主函数
def train_main():
    # 参数设置 训练前请调整这些参数
    batch_size = 256  # 批大小
    num_epochs = 33  # 迭代轮数
    learning_rate = 0.0001  # 学习率
    max_length = 14  # 截断长度
    train_test_ratio = 0.95  # 数据集中训练集与测试集数据量比例
    dataset_filepath = 'pinyin2hanzi.csv'  # 数据集CSV文件路径
    model_config = {  # 模型配置参数
        'd_model': 512,  # 词嵌入维度
        'nhead': 16,  # 多头注意力层注意力头数
        'num_encoder_layers': 8,  # Transformer编码器块数
        'num_decoder_layers': 6,  # Transformer解码器块数
        'dim_feedforward': 1024,  # Transformer前馈层维度
        'dropout': 0.07  # dropout比例
    }

    # 加载数据集
    dataset = PinyinHanziDataset(dataset_filepath, max_length=max_length)

    # 分割训练集和测试集
    train_size = int(train_test_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型包装器
    transformer = PinyinHanziTransformer(
        model=TransformerModel(
            pinyin_vocab_size=len(dataset.pinyin_vocab),
            hanzi_vocab_size=len(dataset.hanzi_vocab),
            **model_config
        ).to(device),
        dataset=dataset,
        config=model_config
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.hanzi2idx['<pad>'])
    optimizer = optim.Adam(transformer.model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.41)

    # 训练循环
    train_losses = []
    test_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_model(transformer.model, train_loader, optimizer, criterion, epoch)
        test_loss = evaluate_model(transformer.model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        scheduler.step()
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

        # 保存整个模型到当前目录（包括词汇表等信息）
        # if epoch % 7 == 0 or epoch == num_epochs:
        transformer.save(f"pinyin2hanzi_transformer_epoch{epoch}.pth")

    # 绘制损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')


# 7. 模型推理主函数
def use_main():
    transformer = PinyinHanziTransformer.load("pinyin2hanzi_transformer.pth", device=str(device))
    result = transformer.predict("hong2 yan2 bo2 ming4")  # 应当输出：红颜薄命
    print("预测结果: ", result)


if __name__ == "__main__":
    # train_main()  # 解除注释、修改参数，运行代码以开始训练
    use_main()  # 解除注释以使用模型
